from django.urls import path, include
from rest_framework import routers
from api import views

router = routers.DefaultRouter()
router.register('programmers', views.ProgrammerViewSet)

urlpatterns = [
    path('', include(router.urls)),
    path('process-image/', views.ProcesImages.process_image, name='process_image'),
    path('shape-image/', views.ProcesImages.get_shape, name='shape_image'),
    path('process-return/', views.process_and_return_image_url, name='process-and-return')
]
