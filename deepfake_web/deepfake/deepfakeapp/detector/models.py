from django.db import models
from django.conf import settings
from django.contrib.auth.models import AbstractUser

class User(AbstractUser):
    pass

class VideoLabel(models.Model):
    video_file = models.FileField(upload_to="videos/")
    uploaded_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True)
    label = models.CharField(max_length=20, choices=[('Real','Real'),('Deepfake','Deepfake'),('Unknown','Unknown')])
    confidence = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.video_file.name} - {self.label}"

class FrameLabel(models.Model):
    video = models.ForeignKey(VideoLabel, on_delete=models.CASCADE, related_name="frames")
    frame_file = models.ImageField(upload_to="frames/")
    label = models.CharField(max_length=20, choices=[('Real','Real'),('Deepfake','Deepfake'),('Unclear','Unclear')])
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.frame_file.name} - {self.label}"
