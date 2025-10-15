# dashboard/tasks.py
import sys
import asyncio
import shutil
import time
import re
import traceback
from io import StringIO
from pathlib import Path

import pandas as pd
from django.conf import settings
from django.utils import timezone

from nbclient import NotebookClient
import nbformat

from .models import UploadJob, Patient, Measurement


# --- pandas 2.x compatibility: restore DataFrame.append for old notebooks ---
if not hasattr(pd.DataFrame, "append"):
    def _append(self, other, *args, **kwargs):
        import pandas as _pd
        return _pd.concat([self, other], axis=0, **kwargs)
    pd.DataFrame.append = _append
# ---------------------------------------------------------------------------


def _parse_intensity_db(s):
    """Extract the first integer from an Intensity string; None if nothing found."""
    if s is None:
        return None
    s = str(s).strip()
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else None


def _exec_notebook(nb_path: Path, work_dir: Path, timeout=3600):
    """
    Execute a Jupyter notebook reliably on Windows by:
      - setting WindowsSelectorEventLoopPolicy
      - running the async execution via asyncio.run(...)
      - falling back to client.execute() if async API isn't present
    """
    try:
        if sys.platform.startswith("win"):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

    nb = nbformat.read(nb_path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(work_dir)}},  # cwd for relative paths in the notebook
    )

    try:
        coro = client.async_execute()  # modern nbclient
        asyncio.run(coro)
    except AttributeError:
        client.execute()
    except RuntimeError:
        # Already in a loop -> allow re-entrancy then run
        try:
            import nest_asyncio  # type: ignore
            nest_asyncio.apply()
            loop = asyncio.get_event_loop()
            loop.run_until_complete(client.async_execute())
        except Exception:
            client.execute()


def run_job(job_id):
    job = UploadJob.objects.get(pk=job_id)
    job.status = "RUNNING"
    job.log_text = ""
    job.save(update_fields=["status", "log_text"])

    log = StringIO()
    try:
        start_all = time.perf_counter()

        base_dir = Path(settings.BASE_DIR)
        media_root = Path(settings.MEDIA_ROOT)

        # Per-job working folders
        job_dir = media_root / "jobs" / str(job.id)
        work_dir = job_dir / "work"
        raw_dir = job_dir / "raw_dataset"
        out_dir = job_dir / "processed_dataset_with_smoothing"
        for d in (work_dir, raw_dir, out_dir):
            d.mkdir(parents=True, exist_ok=True)

        # 1) Copy Excel where the preprocessing notebook expects it
        t0 = time.perf_counter()
        src_excel = Path(job.file.path)
        dst_excel = raw_dir / "Cortical_waveforms.xlsx"
        shutil.copy2(src_excel, dst_excel)
        log.write(f"Copied Excel to {dst_excel}  (+{time.perf_counter()-t0:.1f}s)\n")

        # 2) Execute PREPROCESSING notebook (AS-IS)
        pre_nb = base_dir / "processing" / "dataset_preparation_with_smoothing.ipynb"
        log.write(f"Executing notebook: {pre_nb}\n")
        t0 = time.perf_counter()
        _exec_notebook(pre_nb, work_dir, timeout=3600)
        log.write(f"Preprocessing finished  (+{time.perf_counter()-t0:.1f}s)\n")

        # 3) CSV → DB (import smoothed CSVs into Measurement)
        time_cols = [str(i) for i in range(0, 501, 2)]
        created_total = 0

        for hz in (500, 1000, 2000, 4000):
            f = out_dir / f"{hz}Hz.csv"
            if not f.exists():
                log.write(f"Missing output: {f}\n")
                continue

            t_file = time.perf_counter()
            df = pd.read_csv(f)

            # Which time columns are present?
            cols_present = [c for c in time_cols if c in df.columns]
            if not cols_present:
                cols_present = [c for c in df.columns if str(c).isdigit()]
            if not cols_present:
                log.write(f"{f.name}: no numeric time columns found; skipping file.\n")
                continue

            ok, skipped = 0, 0
            batch = []

            for _, row in df.iterrows():
                patient_code = str(row.get("Patient_ID", "")).strip()
                if not patient_code:
                    skipped += 1
                    continue

                ear = str(row.get("Ear", "")).strip().title()
                if ear not in ("Left", "Right"):
                    ear = "Left"

                intensity_db = _parse_intensity_db(row.get("Intensity"))
                if intensity_db is None:
                    skipped += 1
                    continue

                try:
                    vals = (
                        pd.to_numeric(row[cols_present], errors="coerce")
                        .fillna(0.0).astype(float).tolist()
                    )
                except Exception:
                    skipped += 1
                    continue

                patient, _ = Patient.objects.get_or_create(code=patient_code)
                batch.append(Measurement(
                    patient=patient,
                    sheet_hz=hz,
                    ear=ear,
                    intensity_db=intensity_db,
                    timeseries_json=vals,
                    source_job=job,
                ))
                ok += 1

            if batch:
                Measurement.objects.bulk_create(batch, batch_size=1000)

            created_total += ok
            log.write(f"Imported {ok} rows from {f.name} (skipped {skipped})  (+{time.perf_counter()-t_file:.1f}s)\n")

        # >>> NEW: ensure labeled_images_* folders exist for the notebook's plt.savefig <<<
        for freq in ("500Hz", "1000Hz", "2000Hz", "4000Hz"):
            (out_dir / f"labeled_images_{freq}").mkdir(parents=True, exist_ok=True)

        # 4) Execute LABEL notebook (AS-IS)
        label_nb = base_dir / "processing" / "automated_label_generation_with_smoothing.ipynb"
        log.write(f"Executing notebook: {label_nb}\n")
        t0 = time.perf_counter()
        _exec_notebook(label_nb, work_dir, timeout=3600)
        log.write(f"Labeling finished  (+{time.perf_counter()-t0:.1f}s)\n")

        # 5) Ingest labeled CSVs and link labeled images to measurements
        updated = 0
        for freq in ("500Hz", "1000Hz", "2000Hz", "4000Hz"):
            f = out_dir / f"labeled_data_{freq}.csv"
            hz = int(freq.replace("Hz", ""))

            if not f.exists():
                log.write(f"Missing labeled output: {f}\n")
                continue

            df = pd.read_csv(f)

            for _, row in df.iterrows():
                code = str(row.get("Patient_ID", "")).strip()
                intensity_str = str(row.get("Intensity", "")).strip()
                intensity_db = _parse_intensity_db(intensity_str)

                if not code or intensity_db is None:
                    continue

                label_val = row.get("label", None)
                abs_diff_val = row.get("absolute_difference", None)

                label = int(label_val) if pd.notna(label_val) else None
                abs_diff = float(abs_diff_val) if pd.notna(abs_diff_val) else None

                # Notebook saves: ../processed_dataset_with_smoothing/labeled_images_<freq>/<code>_<Intensity>.png
                img_name = f"{code}_{intensity_str}.png"
                rel_path = Path("jobs") / str(job.id) / "processed_dataset_with_smoothing" / f"labeled_images_{freq}" / img_name
                img_abs = media_root / rel_path

                kwargs = {"label": label, "abs_diff": abs_diff}
                if img_abs.exists():
                    kwargs["labeled_image_rel"] = str(rel_path)

                q = Measurement.objects.filter(
                    source_job=job,
                    patient__code=code,
                    sheet_hz=hz,
                    intensity_db=intensity_db,
                )
                updated += q.update(**kwargs)

        log.write(f"Updated {updated} measurements with labels/images.\n")

        total_time = time.perf_counter() - start_all
        log.write(f"Total job time: {total_time:.1f}s; imported rows: {created_total}\n")

        job.status = "SUCCESS"
        job.finished_at = timezone.now()
        job.log_text = log.getvalue()
        job.save(update_fields=["status", "finished_at", "log_text"])

    except Exception as e:
        log.write("\nERROR:\n")
        log.write("".join(traceback.format_exception(type(e), e, e.__traceback__)))
        job.status = "FAILED"
        job.finished_at = timezone.now()
        job.log_text = log.getvalue()
        job.save(update_fields=["status", "finished_at", "log_text"])
