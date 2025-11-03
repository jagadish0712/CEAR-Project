# cortexdash/urls.py
from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    # Mount the dashboard app at the site root with namespace "dashboard"
    path("", include(("dashboard.urls", "dashboard"), namespace="dashboard")),
]
