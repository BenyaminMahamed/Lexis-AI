from django.urls import path 
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('upload/', views.upload_paper, name='upload_paper'),
    path('paper/<int:pk>/', views.paper_detail, name='paper_detail'),
    path('api/ask/', views.ask, name='ask'),  # Updated to match frontend call
    path('delete/<int:pk>/', views.delete_paper, name='delete_paper'),
]