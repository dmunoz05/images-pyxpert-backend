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

def process_and_return_image_url(request):
    # Procesar la imagen
    image_url = request.GET['image_url']
    # Por ejemplo, cargar una imagen y aplicar un filtro
    req = urllib.request.urlopen(image_url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    image = cv2.imdecode(arr, -1)
    # Realizar manipulaciones con OpenCV, por ejemplo, cambiar a escala de grises
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Guardar la imagen procesada en el sistema de archivos del servidor
    processed_image_path = os.path.join(settings.MEDIA_ROOT, 'processed_image.jpg')
    cv2.imwrite(processed_image_path, gray_image)

    # Construir la URL de la imagen procesada
    processed_image_url = settings.MEDIA_URL + 'processed_image.jpg'

    # Devolver la URL de la imagen procesada como parte de la respuesta JSON
    return JsonResponse({'processed_image_url': processed_image_url})



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

                # imageFinal = cv2.imshow(gray_image)

                # Crear un objeto BytesIO
                buffer = BytesIO()
                # Convertir la imagen a BytesIO
                Image.fromarray(gray_image).save(buffer, format='PNG')

                # Obtener los datos de la imagen
                image_data = buffer.getvalue()

                #Pasar a base64 para enviar
                encode_gray_image_base64 = base64.b64encode(gray_image)

                # Aquí devuelvo un mensaje JSON indicando que la imagen ha sido procesada
                return HttpResponse(image_data, content_type="image/png")
                # return JsonResponse({'status': 'success', 'processed_image': encode_gray_image_base64.decode('utf-8')})
                # return JsonResponse({'status': 'success', 'processed_image': gray_image.tolist()})
            except Exception as e:
                # Devuelve un mensaje de error si no se puede descargar la imagen
                return JsonResponse({'status': 'error', 'message': 'Error al procesar la imagen: {}'.format(str(e))})
        else:
            # Si no se proporciona el parámetro 'image_url' en la solicitud, devuelve un mensaje de error
            return JsonResponse({'status': 'error', 'message': 'Debe proporcionar la URL de la imagen a procesar.'})
