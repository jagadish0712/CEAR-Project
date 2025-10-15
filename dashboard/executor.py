from django.shortcuts import render

# Create your views here.
from concurrent.futures import ThreadPoolExecutor

# single background worker is enough for MVP
_executor = ThreadPoolExecutor(max_workers=1)

def submit_job(job_id):
    # import inside to avoid circular import
    from .tasks import run_job
    _executor.submit(run_job, job_id)
