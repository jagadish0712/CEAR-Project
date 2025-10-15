# dashboard/models.py
import uuid
from django.db import models


class UploadJob(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    file = models.FileField(upload_to="uploads/")
    status = models.CharField(max_length=16, default="PENDING")  # PENDING/RUNNING/SUCCESS/FAILED
    created_at = models.DateTimeField(auto_now_add=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    log_text = models.TextField(blank=True)

    def __str__(self):
        return f"{self.id} ({self.status})"


class Patient(models.Model):
    code = models.CharField(max_length=100, unique=True)  # e.g., "A54_A54_ID-29"

    def __str__(self):
        return self.code


class Measurement(models.Model):
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE)
    sheet_hz = models.IntegerField()                 # 500 / 1000 / 2000 / 4000
    ear = models.CharField(max_length=5)             # "Left" or "Right"
    intensity_db = models.IntegerField()
    timeseries_json = models.JSONField()             # list of floats for 0..500 ms step 2
    source_job = models.ForeignKey(UploadJob, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)

    # --- NEW fields populated by the labeling notebook ingest step ---
    label = models.IntegerField(null=True, blank=True)              # 0/1
    abs_diff = models.FloatField(null=True, blank=True)             # absolute_difference
    labeled_image_rel = models.CharField(max_length=300, blank=True)  # media-relative PNG path
    # ----------------------------------------------------------------

    def __str__(self):
        return f"{self.patient.code} {self.sheet_hz}Hz {self.ear} {self.intensity_db}dB"
