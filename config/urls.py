from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

import views
from django.urls import path
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('papers.urls')),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

urlpatterns = [
    path('', views.index, name='index'),
    path('upload/', views.upload_paper, name='upload_paper'),
    path('paper/<int:pk>/', views.paper_detail, name='paper_detail'),
    path('paper/<int:pk>/delete/', views.delete_paper, name='delete_paper'),
    path('api/ask/', views.ask, name='ask'),
]