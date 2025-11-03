from django.urls import path
from . import views

app_name = "dashboard"

urlpatterns = [
    # Top-level pages
    path("", views.measurements_list_view, name="measurements"),
    path("measurements/", views.measurements_list_view, name="measurements"),

    # Upload + job status
    path("upload/", views.upload_view, name="upload"),
    path("jobs/<uuid:job_id>/", views.job_detail_view, name="job_detail"),

    # Individual measurement utilities (optional pages)
    path("measurements/<int:pk>/", views.measurement_detail_view, name="measurement_detail"),
    path("measurements/<int:pk>/plot.png", views.measurement_plot_png_view, name="measurement_plot_png"),

    # JSON for Plotly
    path("measurements/<int:pk>/data.json", views.measurement_json_view, name="measurement_json"),

    # Annotation API (single)
    path("api/measurements/<int:pk>/annotate/", views.measurement_annotate_api, name="measurement_annotate_api"),

    # BULK save API  ← used by the Save button
    path("api/measurements/bulk-save/", views.measurements_bulk_save_api, name="measurements_bulk_save_api"),
]
