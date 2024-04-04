import json
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse, HttpResponse
from .serializer import ProgrammerSerializer
from rest_framework import viewsets
from .models import Programmer
from decouple import config
from io import BytesIO
from PIL import Image
from io import StringIO
import urllib.request
import numpy as np
import base64
import cv2
import matplotlib.pyplot as plt

# Create your views here.


# PEDIENTE DE REVISAR FUNCIONALIDAD E INSTALAR LIBRERIAS
import cv2
import os
from django.conf import settings
from django.http import JsonResponse


class ProgrammerViewSet(viewsets.ModelViewSet):
    queryset = Programmer.objects.all()
    serializer_class = ProgrammerSerializer


class ProcessKeys(viewsets.ModelViewSet):
    def get_key_client_id(request):
        key_get = request.GET['key']
        try:
            key_validate = config('KEY_GET_VALIDATE_CLIENT_ID')
            if (key_get == key_validate):
                key = config('KEY_CRYPT_CLIENT_ID')
                iv = config('IV_CRYPT_CLIENT_ID')
                return JsonResponse({'status': 'success', 'message': {'key': key, 'iv': iv}})
            else:
                return JsonResponse({'status': 'error', 'message': 'Clave incorrecta'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': 'Error inesperado'})

    def get_key_ia(request):
        key_get = request.GET['key']
        try:
            key_validate = config('KEY_GET_VALIDATE_IA')
            if (key_get == key_validate):
                key = config('KEY_CRYPT_IA')
                iv = config('IV_CRYPT_IA')
                return JsonResponse({'status': 'success', 'message': {'key': key, 'iv': iv}})
            else:
                return JsonResponse({'status': 'error', 'message': 'Clave incorrecta'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': 'Error inesperado'})


class ProcesImages(viewsets.ModelViewSet):

    def search_contourns_img(request):
        if ('image_url' in request.GET):
            image_url = request.GET['image_url']
            try:

                def draw(mask, color, textColor, img, ruta):
                    contornos, _ = cv2.findContours(
                        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for c in contornos:
                        area = cv2.contourArea(c)
                        if area > 2000:
                            M = cv2.moments(c)
                            if (M["m00"] == 0):
                                M["m00"] = 1
                            x = int(M["m10"]/M["m00"])
                            y = int(M["m01"]/M["m00"])
                            cv2.circle(img, (x, y), 7, (0, 255, 0), -1)
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            cv2.putText(img, textColor, (x+10, y), font,
                                        0.95, (0, 255, 0), 1, cv2.LINE_AA)
                            newContourn = cv2.convexHull(c)
                            cv2.drawContours(Img, [newContourn], 0, color, 3)
                            # Añadir contorno
                            listSingleContourn.append(
                                {"contorno": newContourn, "Ruta": ruta})

                # Procesando la imagen
                RuteImg = 'dorito13.jpg'
                Img = cv2.imread(RuteImg)
                imgGray = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
                Imagen_hsv = cv2.cvtColor(Img, cv2.COLOR_BGR2HSV)
                ImgResize = cv2.resize(np.uint8(imgGray), (50, 50))

                # Detectar contornos
                listSingleContourn = []

                # Colores HSV
                # Naranja
                orangeBajo = np.array([5, 100, 20], np.uint8)
                orangeAlto = np.array([15, 255, 255], np.uint8)

                # Amarillo
                amarilloBajo = np.array([15, 100, 20], np.uint8)
                amarilloAlto = np.array([45, 255, 255], np.uint8)

                # Leer imagen
                frameHSV = cv2.cvtColor(Img, cv2.COLOR_BGR2HSV)

                # Mascara naranja y amarillo
                maskOrange = cv2.inRange(frameHSV, orangeBajo, orangeAlto)
                maskYellow = cv2.inRange(frameHSV, amarilloBajo, amarilloAlto)
                maskOrangeAndYellow = cv2.add(maskOrange, maskYellow)

                # Dibujamos los contornos
                draw(maskOrangeAndYellow, (0, 165, 255), 'Centro', Img, RuteImg)

                # Procesar imagen
                # plt.imshow(Img, cmap='gray')
                # plt.title('Imagen en escala de grises')
                # plt.axis('off')
                # plt.show()

                return JsonResponse({'status': 'success', 'image': Img})
            except Exception as e:
                return JsonResponse({'status': 'error', 'message': 'Error al procesar la imagen: {}'.format(str(e))})

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

    def get_rgb_google(request):
        if ('image_url' in request.GET):
            image_url = request.GET['image_url']
            try:
                req = urllib.request.urlopen(image_url)
                arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
                image = cv2.imdecode(arr, -1)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return JsonResponse({'status': 'success', 'image': image})
            except Exception as e:
                return JsonResponse({'status': 'error', 'message': 'Error al procesar la imagen: {}'.format(str(e))})

    def process_image_google(request):
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

    @csrf_exempt
    def process_image_pc(request):
        if request.method == 'POST':
            try:
                # Leer la imagen del cuerpo de la solicitud
                body_unicode = request.body.decode('utf-8')
                body_data = json.loads(body_unicode)
                image_base64 = body_data.get('image')
                if image_base64:
                    decode_image = base64.b64decode(image_base64)

                    # Decodificar la imagen usando OpenCV
                    im_arr = np.frombuffer(decode_image, dtype=np.uint8)
                    image = cv2.imdecode(im_arr, -1)

                    # Cambiar a gris
                    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                    buffer = BytesIO()
                    # Convertir la imagen a BytesIO
                    Image.fromarray(gray_image).save(buffer, format='PNG')

                    # Obtener los datos de la imagen
                    image_data = buffer.getvalue()

                    # Devolver los datos de la imagen como respuesta
                    return HttpResponse(image_data, content_type="image/png")
                else:
                    # Devuelve un mensaje de error si no se proporciona una imagen
                    return JsonResponse({'status': 'error', 'message': 'Debe proporcionar una imagen.'})
            except Exception as e:
                # Devuelve un mensaje de error si no se puede procesar la imagen
                return JsonResponse({'status': 'error', 'message': 'Error al procesar la imagen: {}'.format(str(e))})
        else:
            # Devuelve un mensaje de error si no se recibe una solicitud POST
            return JsonResponse({'status': 'error', 'message': 'Se requiere una solicitud POST para procesar la imagen.'})
