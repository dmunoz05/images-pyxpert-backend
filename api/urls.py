from django.urls import path, include
from rest_framework import routers
from api import views

router = routers.DefaultRouter()
router.register('programmers', views.ProgrammerViewSet)

urlpatterns = [
    path('', include(router.urls)),
    path('key-vi/', views.ProcessKeys.get_keys, name='get_key_and_vi'),
    path('process-image/', views.ProcesImages.process_image, name='process_image'),
    path('shape-image/', views.ProcesImages.get_shape, name='shape_image'),
]
