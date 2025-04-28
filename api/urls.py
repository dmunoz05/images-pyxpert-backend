from django.urls import path, include
from rest_framework import routers
from api import views

router = routers.DefaultRouter()

urlpatterns = [
    path('', include(router.urls)),
    path('key-vi-client-id/', views.ProcessKeys.get_key_client_id, name='get_key_and_vi'),
    path('key-vi-ia/', views.ProcessKeys.get_key_ia, name='get_key_and_vi'),
    path('process-black-white-image-google/', views.ProcessImages.process_bw_image_google, name='process_bw_image_google'),
    path('process-black-white-image-pc/', views.ProcessImages.process_bw_image_pc, name='process_bw_image_pc'),
    path('process-shape-image/', views.ProcessImages.get_shape, name='shape_image'),
    path('process-search-contourn-pc/', views.ProcessImages.search_contourns_img_pc, name='process_search_contourn'),
    path('process-search-contourn-google/', views.ProcessImages.search_contourns_img_google, name='process_search_contourn'),
    path('process-model-pdc-scan/', views.ProcessImages.model_search_DPC, name='model_pdc_scan'),
    path('process-characteristic-image-google/', views.ProcessImages.characteristic_image_google, name='process_characteristic_image_google'),
    path('process-characteristic-image-pc/', views.ProcessImages.characteristic_image_pc, name='process_characteristic_image_pc'),
    path('process-image-qr-pc/', views.ProcessImages.process_qr_image_pc, name='process_qr_image_pc'),
    path('process-model-listening/', views.ProcessAudio.process_model_listening, name='process_model_listening'),
    path('process-model-listening/', views.ProcessAudio.process_model_listening, name='process_model_listening'),
    path('process-app-io/', views.ProcessApplicationIO.process_app, name='process_app'),
]
