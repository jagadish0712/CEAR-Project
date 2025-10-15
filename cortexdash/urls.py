from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path("admin/", admin.site.urls),

    # Include the dashboard app with namespace "dashboard"
    path("", include(("dashboard.urls", "dashboard"), namespace="dashboard")),
]

# (optional) serve media in DEBUG
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
