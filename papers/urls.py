from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views
from .api_views import PaperViewSet

router = DefaultRouter()
router.register(r'papers', PaperViewSet, basename='paper')

urlpatterns = [
    path('api/', include(router.urls)),
    path('', views.index, name='index'),
    path('upload/', views.upload_paper, name='upload_paper'),
    path('paper/<int:pk>/', views.paper_detail, name='paper_detail'),
    path('api/ask/', views.ask, name='ask'),
    path('delete/<int:pk>/', views.delete_paper, name='delete_paper'),
]
