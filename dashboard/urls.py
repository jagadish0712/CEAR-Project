# dashboard/urls.py
from django.urls import path
from . import views

app_name = "dashboard"

urlpatterns = [
    path("upload/", views.upload_view, name="upload"),
    path("jobs/<int:job_id>/", views.job_detail_view, name="job_detail"),

    # list + data
    path("measurements/", views.measurements_list_view, name="measurements"),
    path("measurements/<int:pk>/data.json", views.measurement_json_view, name="measurement_json"),

    # optional detail/png
    path("measurements/<int:pk>/", views.measurement_detail_view, name="measurement_detail"),
    path("measurements/<int:pk>/plot.png", views.measurement_plot_png_view, name="measurement_plot_png"),

    # annotate single (if you already use this)
    path("measurements/<int:pk>/annotate/", views.measurement_annotate_api, name="measurement_annotate_api"),

    # NEW: bulk save (one button)
    path("measurements/bulk-save/", views.measurements_bulk_save_api, name="measurements_bulk_save_api"),
]