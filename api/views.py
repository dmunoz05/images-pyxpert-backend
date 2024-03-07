from .serializer import ProgrammerSerializer
from django.http import JsonResponse, HttpResponse
from rest_framework import viewsets
from .models import Programmer
from io import BytesIO
from PIL import Image
import urllib.request
import numpy as np
import base64
import cv2

# Create your views here.


#PEDIENTE DE REVISAR FUNCIONALIDAD E INSTALAR LIBRERIAS
import cv2
import os
from django.conf import settings
from django.http import JsonResponse


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
        if 'image_url' in request.GET:
            image_url = request.GET['image_url']

            try:
                # Leer imagen
                req = urllib.request.urlopen(image_url)
                arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
                image = cv2.imdecode(arr, -1)

                # Cambiar a gris
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                buffer = BytesIO()
                # Convertir la imagen a BytesIO
                Image.fromarray(gray_image).save(buffer, format='PNG')

                # Obtener los datos de la imagen
                image_data = buffer.getvalue()

                # Aquí devuelvo un mensaje JSON indicando que la imagen ha sido procesada
                return HttpResponse(image_data, content_type="image/png")
            except Exception as e:
                # Devuelve un mensaje de error si no se puede descargar la imagen
                return JsonResponse({'status': 'error', 'message': 'Error al procesar la imagen: {}'.format(str(e))})
        else:
            # Si no se proporciona el parámetro 'image_url' en la solicitud, devuelve un mensaje de error
            return JsonResponse({'status': 'error', 'message': 'Debe proporcionar la URL de la imagen a procesar.'})
