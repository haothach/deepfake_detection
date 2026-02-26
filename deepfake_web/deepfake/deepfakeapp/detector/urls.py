from django.urls import path
from . import views
from django.contrib import admin


urlpatterns = [
    path('', views.home, name='home'),
    path('admin/', admin.site.urls),
    path('about/', views.about, name="about"),
    path('contact/', views.contact, name="contact"),
    path('detect/', views.detect, name="detect"),
]
