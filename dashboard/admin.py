from django.contrib import admin

# Register your models here.
from django.contrib import admin
from .models import UploadJob, Patient, Measurement

@admin.register(UploadJob)
class UploadJobAdmin(admin.ModelAdmin):
    list_display = ("id", "status", "created_at", "finished_at")
    readonly_fields = ("id", "created_at", "finished_at", "status", "log_text")

@admin.register(Patient)
class PatientAdmin(admin.ModelAdmin):
    search_fields = ("code",)

@admin.register(Measurement)
class MeasurementAdmin(admin.ModelAdmin):
    list_display = ("id", "patient", "sheet_hz", "ear", "intensity_db", "source_job")
    list_filter = ("sheet_hz", "ear", "intensity_db")
    search_fields = ("patient__code",)
