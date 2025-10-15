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
from .state_store import get_state, update_state


# -------------------------------
# Upload & Job detail
# -------------------------------

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


def job_detail_view(request: HttpRequest, job_id: int):
    job = get_object_or_404(UploadJob, pk=job_id)
    return render(request, "dashboard/job_detail.html", {"job": job})


# -------------------------------
# Measurements list (Plotly cards)
# -------------------------------

def measurements_list_view(request: HttpRequest):
    qs = Measurement.objects.select_related("patient").all()

    # Filters (Ear removed)
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
        .order_by("code")
        .values_list("code", flat=True)
        .distinct()
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


# -------------------------------
# Detail page + PNG (unchanged)
# -------------------------------

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


# -------------------------------
# JSON for Plotly (reads saved state)
# -------------------------------

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

    state = get_state(pk)

    payload = {
        "id": m.id,
        "patient": m.patient.code,
        "sheet_hz": int(m.sheet_hz) if m.sheet_hz is not None else None,
        "ear": m.ear,
        "intensity_db": int(m.intensity_db) if m.intensity_db is not None else None,
        "timeseries": ts,
        "manual_valley_idx": state.get("valley_idx", getattr(m, "manual_valley_idx", None)),
        "manual_peak_idx": state.get("peak_idx", getattr(m, "manual_peak_idx", None)),
        "label": state.get("label", int(m.label) if m.label is not None else None),
        "abs_diff": state.get("delta", float(m.abs_diff) if m.abs_diff is not None else None),
    }
    return JsonResponse(payload)


# -------------------------------
# Single annotate API (optional)
# -------------------------------

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

    cur = get_state(m.id)
    mv = cur.get("valley_idx")
    mp = cur.get("peak_idx")
    if kind == "valley":
        mv = idx
    else:
        mp = idx

    if mv is None:
        mv = 0
    if mp is None:
        mp = mv

    delta = float(abs(y[mp] - y[mv]))
    label_now = 1 if delta <= eff_threshold else 0

    update_state(m.id, valley_idx=mv, peak_idx=mp, delta=delta, label=label_now)

    try:
        if hasattr(m, "manual_valley_idx"): m.manual_valley_idx = mv
        if hasattr(m, "manual_peak_idx"):   m.manual_peak_idx = mp
        if hasattr(m, "abs_diff"):          m.abs_diff = delta
        if hasattr(m, "label"):             m.label = label_now
        m.save()
    except Exception:
        pass

    return JsonResponse(
        {"ok": True, "valley_idx": mv, "peak_idx": mp, "delta": delta, "label": label_now}
    )


# -------------------------------
# Bulk Save API (one button)
# -------------------------------

@require_POST
def measurements_bulk_save_api(request: HttpRequest):
    try:
        body = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"ok": False, "error": "Invalid JSON"}, status=400)

    try:
        th = float(body.get("th", 1.0))
    except Exception:
        th = 1.0
    th = max(0.0, min(2.0, th))
    eff_threshold = th * 30000.0

    items = body.get("items") or []
    if not isinstance(items, list):
        return JsonResponse({"ok": False, "error": "items must be a list"}, status=400)

    results = []
    for it in items:
        try:
            mid = int(it.get("id"))
            v   = it.get("valley_idx")
            p   = it.get("peak_idx")
            if v is None or p is None:
                continue
            m = Measurement.objects.select_related("patient").get(pk=mid)
        except Exception:
            continue

        raw = m.timeseries_json or []
        y: list[float] = []
        for vv in raw:
            try:
                f = float(vv)
            except Exception:
                f = 0.0
            if not math.isfinite(f):
                f = 0.0
            y.append(f)

        n = len(y)
        if n == 0:
            continue

        v = max(0, min(n - 1, int(v)))
        p = max(0, min(n - 1, int(p)))

        delta = float(abs(y[p] - y[v]))
        label_now = 1 if delta <= eff_threshold else 0

        update_state(mid, valley_idx=v, peak_idx=p, delta=delta, label=label_now)

        try:
            changed_fields = []
            if hasattr(m, "manual_valley_idx"):
                m.manual_valley_idx = v
                changed_fields.append("manual_valley_idx")
            if hasattr(m, "manual_peak_idx"):
                m.manual_peak_idx = p
                changed_fields.append("manual_peak_idx")
            if hasattr(m, "abs_diff"):
                m.abs_diff = delta
                changed_fields.append("abs_diff")
            if hasattr(m, "label"):
                m.label = label_now
                changed_fields.append("label")
            if changed_fields:
                m.save(update_fields=list(set(changed_fields)))
        except Exception:
            pass

        results.append({"id": mid, "delta": delta, "label": label_now, "valley_idx": v, "peak_idx": p})

    return JsonResponse({"ok": True, "results": results})
