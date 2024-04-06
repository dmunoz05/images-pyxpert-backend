from django.urls import path, include
from rest_framework import routers
from api import views

router = routers.DefaultRouter()
router.register('programmers', views.ProgrammerViewSet)

urlpatterns = [
    path('', include(router.urls)),
    path('key-vi-client-id/', views.ProcessKeys.get_key_client_id, name='get_key_and_vi'),
    path('key-vi-ia/', views.ProcessKeys.get_key_ia, name='get_key_and_vi'),
    path('process-image-google/', views.ProcessImages.process_image_google, name='process_image_google'),
    path('process-image-pc/', views.ProcessImages.process_image_pc, name='process_image_pc'),
    path('shape-image/', views.ProcessImages.get_shape, name='shape_image'),
    path('process-search-contourn/', views.ProcessImages.search_contourns_with_color_img, name='process_search_contourn'),
]
