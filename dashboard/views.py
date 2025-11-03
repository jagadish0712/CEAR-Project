from __future__ import annotations

import io
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from django.conf import settings
from django.db import transaction
from django.http import (
    HttpResponse,
    JsonResponse,
    FileResponse,
    HttpRequest,
)
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST

from .executor import submit_job
from .forms import UploadForm
from .models import UploadJob, Patient, Measurement


# ---------------- Upload ----------------

def upload_view(request: HttpRequest):
    if request.method == "POST":
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            with transaction.atomic():
                job = UploadJob.objects.create(
                    file=form.cleaned_data["file"], status="PENDING"
                )
            submit_job(job.id)
            return redirect("dashboard:job_detail", job_id=job.id)
    else:
        form = UploadForm()
    return render(request, "dashboard/upload.html", {"form": form})


def job_detail_view(request: HttpRequest, job_id):
    job = get_object_or_404(UploadJob, pk=job_id)
    return render(request, "dashboard/job_detail.html", {"job": job})


# -------------- Measurements list --------------

def measurements_list_view(request: HttpRequest):
    qs = Measurement.objects.select_related("patient").all()

    # Filters (Ear removed as requested)
    patient_code = (request.GET.get("patient_code") or "").strip()
    sheet_hz = request.GET.get("sheet_hz") or ""
    intensity_db = request.GET.get("intensity_db") or ""

    # Threshold factor (0..2)
    try:
        th = float(request.GET.get("th", "1") or "1")
    except ValueError:
        th = 1.0
    th = max(0.0, min(2.0, th))

    if sheet_hz.isdigit():
        qs = qs.filter(sheet_hz=int(sheet_hz))

    if patient_code:
        if Patient.objects.filter(code=patient_code).exists():
            qs = qs.filter(patient__code=patient_code)
        else:
            qs = qs.filter(patient__code__icontains=patient_code)

    if intensity_db.isdigit():
        qs = qs.filter(intensity_db=int(intensity_db))

    MAX_ROWS = 40 if patient_code else 30
    items = list(qs.order_by("-id")[:MAX_ROWS])

    patient_list = (
        Patient.objects.filter(measurement__isnull=False)
        .order_by("code").values_list("code", flat=True).distinct()
    )

    return render(
        request,
        "dashboard/measurements_list.html",
        {
            "items": items,
            "patient_list": patient_list,
            "th": th,
            "filters": {
                "sheet_hz": sheet_hz,
                "patient_code": patient_code,
                "intensity_db": intensity_db,
            },
        },
    )


# -------------- Detail (optional) --------------

def measurement_detail_view(request: HttpRequest, pk: int):
    m = get_object_or_404(Measurement, pk=pk)
    return render(request, "dashboard/measurement_detail.html", {"m": m})


def measurement_plot_png_view(request: HttpRequest, pk: int):
    m = get_object_or_404(Measurement, pk=pk)

    try:
        th = float(request.GET.get("th", "1") or "1")
    except ValueError:
        th = 1.0
    th = max(0.0, min(2.0, th))
    eff_threshold = th * 30000.0

    if getattr(m, "labeled_image_rel", None) and abs(th - 1.0) < 1e-9:
        img_abs = Path(settings.MEDIA_ROOT) / m.labeled_image_rel
        if img_abs.exists():
            return FileResponse(open(img_abs, "rb"), content_type="image/png")

    raw = m.timeseries_json or []
    y = []
    for v in raw:
        try:
            f = float(v)
        except Exception:
            f = 0.0
        if not math.isfinite(f):
            f = 0.0
        y.append(f)
    x = list(range(0, 2 * len(y), 2))

    def draw_plain(title_suffix=""):
        fig = plt.figure()
        if y:
            plt.plot(x, y)
        title = f"{m.patient.code} | {m.sheet_hz} Hz | {m.ear} | {m.intensity_db} dB"
        if title_suffix:
            title += f" | {title_suffix}"
        plt.title(title)
        plt.xlabel("Time (ms)")
        plt.ylabel("Amplitude")
        out = io.BytesIO()
        fig.savefig(out, format="png", bbox_inches="tight")
        plt.close(fig)
        out.seek(0)
        return HttpResponse(out.read(), content_type="image/png")

    if not y:
        return draw_plain()

    try:
        def ms_to_idx(ms: int) -> int:
            idx = int(ms // 2)
            return max(0, min(len(y) - 1, idx))

        valley_start = ms_to_idx(45)
        valley_end = ms_to_idx(150)
        search_end = ms_to_idx(300)

        seg = np.asarray(y[valley_start:valley_end + 1], dtype=float)
        if seg.size == 0:
            return draw_plain()

        valley_rel = int(np.nanargmin(seg))
        valley_idx = valley_start + valley_rel

        right_start = min(valley_idx + 1, len(y) - 1)
        right_end = min(search_end, len(y) - 1)
        right_seg = np.asarray(y[right_start:right_end + 1], dtype=float)
        if right_seg.size == 0:
            return draw_plain()

        peak_rel = int(np.nanargmax(right_seg))
        peak_idx = right_start + peak_rel

        delta = float(abs(y[peak_idx] - y[valley_idx]))
        label_now = 1 if delta <= eff_threshold else 0

        fig = plt.figure()
        plt.plot(x, y)
        plt.scatter(x[valley_idx], y[valley_idx], c="red", zorder=5, label="First Valley")
        plt.scatter(x[peak_idx], y[peak_idx], c="green", zorder=5, label="Highest Peak After Valley")
        plt.legend(loc="lower right")

        title = (
            f"{m.patient.code} | {m.sheet_hz} Hz | {m.ear} | {m.intensity_db} dB\n"
            f"Δ={delta:.0f}, threshold={eff_threshold:.0f} (×{th:.2f}) → label={label_now}"
        )
        plt.title(title)
        plt.xlabel("Time (ms)")
        plt.ylabel("Amplitude")

        out = io.BytesIO()
        fig.savefig(out, format="png", bbox_inches="tight")
        plt.close(fig)
        out.seek(0)
        return HttpResponse(out.read(), content_type="image/png")

    except Exception:
        return draw_plain()


# -------------- JSON for Plotly --------------

def measurement_json_view(request: HttpRequest, pk: int):
    m = get_object_or_404(Measurement, pk=pk)

    raw = m.timeseries_json or []
    ts: list[float] = []
    for v in raw:
        try:
            f = float(v)
        except Exception:
            f = 0.0
        if not math.isfinite(f):
            f = 0.0
        ts.append(f)

    payload = {
        "id": m.id,
        "patient": m.patient.code,
        "sheet_hz": int(m.sheet_hz) if m.sheet_hz is not None else None,
        "ear": m.ear,
        "intensity_db": int(m.intensity_db) if m.intensity_db is not None else None,
        "timeseries": ts,
        "label": int(m.label) if m.label is not None else None,
        "abs_diff": float(m.abs_diff) if m.abs_diff is not None else None,
        "manual_valley_idx": getattr(m, "manual_valley_idx", None),
        "manual_peak_idx": getattr(m, "manual_peak_idx", None),
        "labeled_image": str(m.labeled_image_rel) if getattr(m, "labeled_image_rel", None) else None,
    }
    return JsonResponse(payload)


# -------------- Single annotate API (optional) --------------

@require_POST
def measurement_annotate_api(request: HttpRequest, pk: int):
    m = get_object_or_404(Measurement, pk=pk)

    try:
        body = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"ok": False, "error": "Invalid JSON body"}, status=400)

    kind = (body.get("kind") or "").strip().lower()
    if kind not in ("valley", "peak"):
        return JsonResponse({"ok": False, "error": "kind must be 'valley' or 'peak'"}, status=400)

    try:
        idx = int(body.get("index"))
    except Exception:
        return JsonResponse({"ok": False, "error": "index must be int"}, status=400)

    try:
        th = float(body.get("th", 1.0))
    except Exception:
        th = 1.0
    th = max(0.0, min(2.0, th))
    eff_threshold = th * 30000.0

    raw = m.timeseries_json or []
    y: list[float] = []
    for v in raw:
        try:
            f = float(v)
        except Exception:
            f = 0.0
        if not math.isfinite(f):
            f = 0.0
        y.append(f)

    n = len(y)
    if n == 0:
        return JsonResponse({"ok": False, "error": "empty timeseries"}, status=400)

    idx = max(0, min(n - 1, idx))

    if hasattr(m, "manual_valley_idx") and kind == "valley":
        m.manual_valley_idx = idx
    if hasattr(m, "manual_peak_idx") and kind == "peak":
        m.manual_peak_idx = idx

    # complete the pair & compute
    def ms_to_idx(ms: int) -> int:
        i = int(ms // 2)
        return max(0, min(n - 1, i))

    mv = getattr(m, "manual_valley_idx", None)
    mp = getattr(m, "manual_peak_idx", None)

    if mv is None:
        vs, ve = ms_to_idx(45), ms_to_idx(150)
        seg = np.asarray(y[vs:ve + 1], dtype=float)
        mv = vs + int(np.nanargmin(seg)) if seg.size > 0 else 0

    if mp is None:
        right_start = min(mv + 1, n - 1)
        right_end = min(ms_to_idx(300), n - 1)
        seg = np.asarray(y[right_start:right_end + 1], dtype=float)
        mp = right_start + int(np.nanargmax(seg)) if seg.size > 0 else mv

    delta = float(abs(y[mp] - y[mv]))
    label_now = 1 if delta <= eff_threshold else 0

    if hasattr(m, "abs_diff"):
        m.abs_diff = delta
    if hasattr(m, "label"):
        m.label = label_now
    try:
        m.save(update_fields=["abs_diff", "label", "manual_valley_idx", "manual_peak_idx"])
    except Exception:
        m.save()

    return JsonResponse({"ok": True, "valley_idx": mv, "peak_idx": mp, "delta": delta, "label": label_now})


# -------------- BULK save API (used by Save button) --------------

@require_POST
def measurements_bulk_save_api(request: HttpRequest):
    """
    Body JSON:
    {
      "items": [
        {"id": 123, "valley_idx": 78, "peak_idx": 156},
        ...
      ]
    }
    Recomputes Δ and label at th (from query string or default 1.0).
    """
    try:
        body = json.loads(request.body.decode("utf-8"))
        items = body.get("items", [])
    except Exception:
        return JsonResponse({"ok": False, "error": "Invalid JSON body"}, status=400)

    try:
        th = float(request.GET.get("th", "1") or "1")
    except ValueError:
        th = 1.0
    th = max(0.0, min(2.0, th))
    eff_threshold = th * 30000.0

    updated = []

    for it in items:
        try:
            mid = int(it.get("id"))
        except Exception:
            continue
        m = Measurement.objects.select_related("patient").filter(pk=mid).first()
        if not m:
            continue

        raw = m.timeseries_json or []
        y: list[float] = []
        for v in raw:
            try:
                f = float(v)
            except Exception:
                f = 0.0
            if not math.isfinite(f):
                f = 0.0
            y.append(f)
        if not y:
            continue

        n = len(y)
        v_idx = it.get("valley_idx")
        p_idx = it.get("peak_idx")
        if v_idx is None or p_idx is None:
            # fallback to the same heuristic
            def ms_to_idx(ms: int) -> int:
                return max(0, min(n - 1, int(ms // 2)))
            vs, ve = ms_to_idx(45), ms_to_idx(150)
            seg = np.asarray(y[vs:ve + 1], dtype=float)
            v_idx = vs + int(np.nanargmin(seg)) if seg.size > 0 else 0
            right_start = min(v_idx + 1, n - 1)
            right_end = min(ms_to_idx(300), n - 1)
            seg2 = np.asarray(y[right_start:right_end + 1], dtype=float)
            p_idx = right_start + int(np.nanargmax(seg2)) if seg2.size > 0 else v_idx

        v_idx = max(0, min(n - 1, int(v_idx)))
        p_idx = max(0, min(n - 1, int(p_idx)))

        delta = float(abs(y[p_idx] - y[v_idx]))
        label_now = 1 if delta <= eff_threshold else 0

        # persist
        if hasattr(m, "manual_valley_idx"):
            m.manual_valley_idx = v_idx
        if hasattr(m, "manual_peak_idx"):
            m.manual_peak_idx = p_idx
        if hasattr(m, "abs_diff"):
            m.abs_diff = delta
        if hasattr(m, "label"):
            m.label = label_now
        try:
            m.save(update_fields=["manual_valley_idx", "manual_peak_idx", "abs_diff", "label"])
        except Exception:
            m.save()

        updated.append({"id": m.id, "valley_idx": v_idx, "peak_idx": p_idx, "delta": delta, "label": label_now})

    return JsonResponse({"ok": True, "updated": updated})
