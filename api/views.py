from .serializer import ProgrammerSerializer
from rest_framework import viewsets
from .models import Programmer
from django.http import JsonResponse
import urllib.request
import numpy as np
import cv2
from django.shortcuts import render
from django.views import View
# Create your views here.


class ProgrammerViewSet(viewsets.ModelViewSet):
    queryset = Programmer.objects.all()
    serializer_class = ProgrammerSerializer

class ProcesImages(viewsets.ModelViewSet):
    #devolver shape
    def get_shape(request):
        if 'image_url' in request.GET:
            image_url = request.GET['image_url']
            try:
                req = urllib.request.urlopen(image_url)
                arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
                image = cv2.imdecode(arr, -1)
                return JsonResponse({'status': 'success', 'shape': image.shape})
            except Exception as e:
                return JsonResponse({'status': 'error', 'message': 'Error al procesar la imagen: {}'.format(str(e))})

    def process_image(request):
        # Verifica si el parámetro 'image_url' está presente en la solicitud
        if 'image_url' in request.GET:
            # Obtiene la URL de la imagen del parámetro 'image_url'
            image_url = request.GET['image_url']

            try:
                # Abre la URL de la imagen y la lee
                req = urllib.request.urlopen(image_url)
                arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
                image = cv2.imdecode(arr, -1)

                # Realiza la manipulación de la imagen aquí utilizando OpenCV
                # Por ejemplo, convertirla a escala de grises
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Aquí devuelvo un mensaje JSON indicando que la imagen ha sido procesada
                return JsonResponse({'status': 'success', 'processed_image': gray_image.tolist()})
            except Exception as e:
                # Devuelve un mensaje de error si no se puede descargar la imagen
                return JsonResponse({'status': 'error', 'message': 'Error al procesar la imagen: {}'.format(str(e))})
        else:
            # Si no se proporciona el parámetro 'image_url' en la solicitud, devuelve un mensaje de error
            return JsonResponse({'status': 'error', 'message': 'Debe proporcionar la URL de la imagen a procesar.'})
