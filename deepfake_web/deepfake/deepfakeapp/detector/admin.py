from django.contrib import admin
from .models import User, VideoLabel, FrameLabel

admin.site.register(VideoLabel)
admin.site.register(FrameLabel)
