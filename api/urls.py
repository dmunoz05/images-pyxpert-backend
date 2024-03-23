from django.urls import path, include
from rest_framework import routers
from api import views

router = routers.DefaultRouter()
router.register('programmers', views.ProgrammerViewSet)

urlpatterns = [
    path('', include(router.urls)),
    path('key-vi/', views.ProcessKeys.get_keys, name='get_key_and_vi'),
    path('process-image-google/', views.ProcesImages.process_image_google, name='process_image_google'),
    path('process-image-pc/', views.ProcesImages.process_image_pc, name='process_image_pc'),
    path('shape-image/', views.ProcesImages.get_shape, name='shape_image'),
]
