from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse, HttpResponse
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit
from rest_framework import viewsets
from django.conf import settings
import matplotlib.pyplot as plt
from decouple import config
from io import StringIO
from io import BytesIO
from PIL import Image
import urllib.request
import numpy as np
import base64
import pickle
import joblib
import json
import cv2
import os


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
            return JsonResponse({'status': 'error', 'message': 'Error al procesar la imagen: {}'.format(str(e))})

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
            return JsonResponse({'status': 'error', 'message': 'Error al procesar la imagen: {}'.format(str(e))})


class ProcessImages(viewsets.ModelViewSet):

    def characteristic_image_google(request):
        if 'image_url' in request.GET:
            image_url = request.GET['image_url']

            try:
                # Leer imagen
                req = urllib.request.urlopen(image_url)
                arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
                result = cv2.imdecode(arr, -1)

                # Procesando la imagen
                g = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
                inverted = cv2.bitwise_not(g)
                _in = inverted > 100
                _bin = cv2.resize(np.uint8(_in), (50, 50))

                # Encontrar los contornos de la imagen
                contours, _ = cv2.findContours(
                    np.uint8(_in), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if len(contours) > 0:
                    # Tomar solo el contorno más grande (puede haber varios)
                    largest_contour = max(contours, key=cv2.contourArea)

                # Calcular área
                area = cv2.contourArea(largest_contour)

                # Calcular perímetro
                perimeter = cv2.arcLength(largest_contour, closed=True)

                # Calcular el centro de masa
                M = cv2.moments(largest_contour)
                center_x = int(M['m10'] / M['m00'])
                center_y = int(M['m01'] / M['m00'])

                # Calcular la circularidad
                circularity = (4 * np.pi * area) / (perimeter ** 2)

                # Calcular la elipticidad
                _, (major_axis, minor_axis), _ = cv2.fitEllipse(
                    largest_contour)
                ellipticity = major_axis / minor_axis

                # Calcular los momentos de Hu
                _hu_moments = np.transpose(cv2.HuMoments(M))  # Siete valores

                # Agregando alto y ancho
                _p1, _p2, _w, _h = cv2.boundingRect(np.uint8(_bin))

                data = {'area': area, 'perimeter': perimeter, 'ellipticity': ellipticity, 'center_x': center_x,
                        'center_y': center_y, 'circularity': circularity, 'Hu_moments': _hu_moments.tolist(), 'Height': _h, 'Width': _w}

                return JsonResponse({'status': 200, 'characteristics': data})
            except Exception as e:
                return JsonResponse({'status': 'error', 'message': 'Error al procesar la imagen: {}'.format(str(e))})

    @csrf_exempt
    def characteristic_image_pc(request):
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
                    _image = cv2.imdecode(im_arr, -1)

                    # Procesando la imagen
                    _gris = cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY)
                    _grisinverted2 = cv2.bitwise_not(_gris)
                    _in = _grisinverted2 > 100
                    _bin = cv2.resize(np.uint8(_in), (50, 50))

                    # Encontrar los contornos de la imagen
                    contours, _ = cv2.findContours(
                        np.uint8(_bin), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    if len(contours) > 0:
                        # Tomar solo el contorno más grande (puede haber varios)
                        largest_contour = max(contours, key=cv2.contourArea)

                    # Calcular área
                    area = cv2.contourArea(largest_contour)

                    # Calcular perímetro
                    perimeter = cv2.arcLength(largest_contour, closed=True)

                    # Calcular el centro de masa
                    M = cv2.moments(largest_contour)
                    center_x = int(M['m10'] / M['m00'])
                    center_y = int(M['m01'] / M['m00'])

                    # Calcular la circularidad
                    circularity = (4 * np.pi * area) / (perimeter ** 2)

                    # Calcular la elipticidad
                    _, (major_axis, minor_axis), _ = cv2.fitEllipse(
                        largest_contour)
                    ellipticity = major_axis / minor_axis

                    # Calcular los momentos de Hu
                    _hu_moments = np.transpose(
                        cv2.HuMoments(M))  # Siete valores

                    # Agregando alto y ancho
                    _p1, _p2, _w, _h = cv2.boundingRect(np.uint8(_bin))

                    data = {'area': area, 'perimeter': perimeter, 'ellipticity': ellipticity, 'center_x': center_x,
                            'center_y': center_y, 'circularity': circularity, 'Hu_moments': _hu_moments.tolist(), 'Height': _h, 'Width': _w}

                return JsonResponse({'status': 200, 'characteristics': data})
            except Exception as e:
                return JsonResponse({'status': 'error', 'message': 'Error al procesar la imagen: {}'.format(str(e))})

    @ csrf_exempt
    def model_search_DPC(request):
        if (request.method == 'POST'):
            try:
                # Leer la imagen del cuerpo de la solicitud
                body_unicode = request.body.decode('utf-8')
                body_data = json.loads(body_unicode)
                image_base64 = body_data.get('image')
                if (image_base64):
                    decode_image = base64.b64decode(image_base64)
                    # Decodificar la imagen usando OpenCV
                    im_arr = np.frombuffer(decode_image, dtype=np.uint8)
                    _image = cv2.imdecode(im_arr, -1)

                    # Obtener el directorio de trabajo actual
                    cwd = os.getcwd()
                    current_directory = os.path.join(
                        cwd, 'models', 'Modelo_knn5.pkl')

                    # Cargar archivos desde el sistema local
                    with open(current_directory, 'rb') as f:
                        # Con el KNN5
                        modelo_entrenado = pickle.load(f)

                        # Con el KNN4
                        # modelo_entrenado = joblib.load(f)

                    # Procesando la imagen
                    _gris = cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY)
                    _grisinverted2 = cv2.bitwise_not(_gris)
                    _in = _grisinverted2 > 100
                    _bin = cv2.resize(np.uint8(_in), (50, 50))

                    # Midiendo el área de las letras (Igual que en el entrenamiento)
                    _x_new = np.zeros((1, 15))

                    # Encontrar los contornos de la imagen
                    contours, _ = cv2.findContours(
                        np.uint8(_bin), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    if (len(contours) > 0):
                        # Tomar solo el contorno más grande (puede haber varios)
                        largest_contour = max(contours, key=cv2.contourArea)

                        # Calcular área
                        area = cv2.contourArea(largest_contour)

                        # Calcular perímetro
                        perimeter = cv2.arcLength(largest_contour, closed=True)

                        # Calcular el centro de masa
                        M = cv2.moments(largest_contour)
                        center_x = int(M['m10'] / M['m00'])
                        center_y = int(M['m01'] / M['m00'])

                        # Calcular la circularidad
                        circularity = (4 * np.pi * area) / (perimeter ** 2)

                        # Calcular la elipticidad
                        _, (major_axis, minor_axis), _ = cv2.fitEllipse(
                            largest_contour)
                        ellipticity = major_axis / minor_axis

                        # Calcular los momentos de Hu
                        _hu_moments = np.transpose(
                            cv2.HuMoments(M))  # Siete valores

                        # Agregando alto y ancho
                        _p1, _p2, _ancho, _alto = cv2.boundingRect(
                            np.uint8(_bin))

                        # Almacenando resultados
                        _x_new[0, 0] = area
                        _x_new[0, 1] = perimeter
                        _x_new[0, 2] = ellipticity
                        _x_new[0, 3] = center_x
                        _x_new[0, 4] = center_y
                        _x_new[0, 5] = circularity
                        _x_new[0, 6:13] = _hu_moments
                        _x_new[0, 13] = _alto
                        _x_new[0, 14] = _ancho

                        # scaler = MinMaxScaler()

                        # _x_fit = scaler.fit(_x_new)

                        # x_fit = scaler.fit_transform(_x_new)
                        # _x_new_normalized = scaler.transform(_x_new)

                        value_predict = modelo_entrenado.predict(_x_new)
                        if value_predict == 0:
                            _messaje_response = 'Estoy reconociendo papas'
                        if value_predict == 1:
                            _messaje_response = 'Estoy reconociendo Doritos'
                        if value_predict == 2:
                            _messaje_response = 'Estoy reconociendo Cheese tris'
                    # Aquí devuelvo un mensaje JSON indicando que la imagen ha sido procesada
                return JsonResponse({'status': 'error', 'message': _messaje_response})
            except Exception as e:
                return JsonResponse({'status': 'error', 'message': 'Error al procesar la imagen: {}'.format(str(e))})

    def search_contourns_img_google(request):
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
                    _img = cv2.imdecode(im_arr, -1)

                    def draw(mask, color, text_color, img):
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
                                cv2.putText(img, text_color, (x+10, y),
                                            font, 0.95, (0, 255, 0), 1, cv2.LINE_AA)
                                new_contourn = cv2.convexHull(c)
                                cv2.drawContours(
                                    _img, [new_contourn], 0, color, 3)

                    # Colores HSV

                    # Rojo
                    red_bajo1 = np.array([0, 100, 20], np.uint8)
                    red_alto1 = np.array([5, 255, 255], np.uint8)
                    red_bajo2 = np.array([175, 100, 20], np.uint8)
                    red_alto2 = np.array([180, 255, 255], np.uint8)

                    # Naranja
                    orange_bajo = np.array([5, 100, 20], np.uint8)
                    orange_alto = np.array([15, 255, 255], np.uint8)

                    # Amarillo
                    amarillo_bajo = np.array([15, 100, 20], np.uint8)
                    amarillo_alto = np.array([45, 255, 255], np.uint8)

                    # Verde
                    verde_bajo = np.array([45, 100, 20], np.uint8)
                    verde_alto = np.array([85, 255, 255], np.uint8)

                    # Azul claro
                    azul_bajo1 = np.array([100, 100, 20], np.uint8)
                    azul_alto1 = np.array([125, 255, 255], np.uint8)

                    # Azul oscuro
                    azul_bajo2 = np.array([125, 100, 20], np.uint8)
                    azul_alto2 = np.array([130, 255, 255], np.uint8)

                    # Morado
                    morado_bajo = np.array([135, 100, 20], np.uint8)
                    morado_alto = np.array([145, 255, 255], np.uint8)

                    # Violeta
                    violeta_bajo = np.array([145, 100, 20], np.uint8)
                    violeta_alto = np.array([170, 255, 255], np.uint8)

                    frame_hsv = cv2.cvtColor(_img, cv2.COLOR_BGR2HSV)
                    # Detectamos los colores

                    # Rojo
                    mask_red1 = cv2.inRange(frame_hsv, red_bajo1, red_alto1)
                    mask_red2 = cv2.inRange(frame_hsv, red_bajo2, red_alto2)
                    mask_red = cv2.add(mask_red1, mask_red2)

                    # Naranja
                    mask_orange = cv2.inRange(
                        frame_hsv, orange_bajo, orange_alto)

                    # Amarillo
                    mask_amarillo = cv2.inRange(
                        frame_hsv, amarillo_bajo, amarillo_alto)

                    # Verde
                    mask_verde = cv2.inRange(frame_hsv, verde_bajo, verde_alto)

                    # Azul
                    mask_azul1 = cv2.inRange(frame_hsv, azul_bajo1, azul_alto1)
                    mask_azul2 = cv2.inRange(frame_hsv, azul_bajo2, azul_alto2)
                    mask_azul = cv2.add(mask_azul1, mask_azul2)

                    # Morado
                    mask_morado = cv2.inRange(
                        frame_hsv, morado_bajo, morado_alto)

                    # Violeta
                    mask_violeta = cv2.inRange(
                        frame_hsv, violeta_bajo, violeta_alto)

                    # Dibujamos los contornos
                    draw(mask_red, (0, 0, 255), 'Rojo', _img)
                    draw(mask_orange, (0, 165, 255), 'Naranja', _img)
                    draw(mask_amarillo, (0, 255, 255), 'Amarillo', _img)
                    draw(mask_verde, (0, 255, 0), 'Verde', _img)
                    draw(mask_azul, (255, 0, 0), 'Azul', _img)
                    draw(mask_morado, (255, 0, 255), 'Morado', _img)
                    draw(mask_violeta, (255, 0, 255), 'Violeta', _img)

                    buffer = BytesIO()

                    # Convertir la imagen a BytesIO
                    Image.fromarray(_img).save(buffer, format='PNG')

                    # Obtener los datos de la imagen
                    image_data = buffer.getvalue()

                    # Aquí devuelvo un mensaje JSON indicando que la imagen ha sido procesada
                return HttpResponse(image_data, content_type="image/png")
            except Exception as e:
                return JsonResponse({'status': 'error', 'message': 'Error al procesar la imagen: {}'.format(str(e))})

    @ csrf_exempt
    def search_contourns_img_pc(request):
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
                    Img = cv2.imdecode(im_arr, -1)

                    def draw(mask, color, textColor, img):
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
                                cv2.putText(img, textColor, (x+10, y),
                                            font, 0.95, (0, 255, 0), 1, cv2.LINE_AA)
                                newContourn = cv2.convexHull(c)
                                cv2.drawContours(
                                    Img, [newContourn], 0, color, 3)

                    # Colores HSV

                    # Rojo
                    redBajo1 = np.array([0, 100, 20], np.uint8)
                    redAlto1 = np.array([5, 255, 255], np.uint8)
                    redBajo2 = np.array([175, 100, 20], np.uint8)
                    redAlto2 = np.array([180, 255, 255], np.uint8)

                    # Naranja
                    orangeBajo = np.array([5, 100, 20], np.uint8)
                    orangeAlto = np.array([15, 255, 255], np.uint8)

                    # Amarillo
                    amarilloBajo = np.array([15, 100, 20], np.uint8)
                    amarilloAlto = np.array([45, 255, 255], np.uint8)

                    # Verde
                    verdeBajo = np.array([45, 100, 20], np.uint8)
                    verdeAlto = np.array([85, 255, 255], np.uint8)

                    # Azul claro
                    azulBajo1 = np.array([100, 100, 20], np.uint8)
                    azulAlto1 = np.array([125, 255, 255], np.uint8)

                    # Azul oscuro
                    azulBajo2 = np.array([125, 100, 20], np.uint8)
                    azulAlto2 = np.array([130, 255, 255], np.uint8)

                    # Morado
                    moradoBajo = np.array([135, 100, 20], np.uint8)
                    moradoAlto = np.array([145, 255, 255], np.uint8)

                    # Violeta
                    violetaBajo = np.array([145, 100, 20], np.uint8)
                    violetaAlto = np.array([170, 255, 255], np.uint8)

                    frameHSV = cv2.cvtColor(Img, cv2.COLOR_BGR2HSV)
                    # Detectamos los colores

                    # Rojo
                    maskRed1 = cv2.inRange(frameHSV, redBajo1, redAlto1)
                    maskRed2 = cv2.inRange(frameHSV, redBajo2, redAlto2)
                    maskRed = cv2.add(maskRed1, maskRed2)

                    # Naranja
                    maskOrange = cv2.inRange(frameHSV, orangeBajo, orangeAlto)

                    # Amarillo
                    maskAmarillo = cv2.inRange(
                        frameHSV, amarilloBajo, amarilloAlto)

                    # Verde
                    maskVerde = cv2.inRange(frameHSV, verdeBajo, verdeAlto)

                    # Azul
                    maskAzul1 = cv2.inRange(frameHSV, azulBajo1, azulAlto1)
                    maskAzul2 = cv2.inRange(frameHSV, azulBajo2, azulAlto2)
                    maskAzul = cv2.add(maskAzul1, maskAzul2)

                    # Morado
                    maskMorado = cv2.inRange(frameHSV, moradoBajo, moradoAlto)

                    # Violeta
                    maskVioleta = cv2.inRange(
                        frameHSV, violetaBajo, violetaAlto)

                    # Dibujamos los contornos
                    draw(maskRed, (0, 0, 255), 'Rojo', Img)
                    draw(maskOrange, (0, 165, 255), 'Naranja', Img)
                    draw(maskAmarillo, (0, 255, 255), 'Amarillo', Img)
                    draw(maskVerde, (0, 255, 0), 'Verde', Img)
                    draw(maskAzul, (255, 0, 0), 'Azul', Img)
                    draw(maskMorado, (255, 0, 255), 'Morado', Img)
                    draw(maskVioleta, (255, 0, 255), 'Violeta', Img)
                    # draw(maskOrangeAndYellow, (0, 165, 255), 'Centro', Img)

                    buffer = BytesIO()

                    # Convertir la imagen a BytesIO
                    Image.fromarray(Img).save(buffer, format='PNG')

                    # Obtener los datos de la imagen
                    image_data = buffer.getvalue()

                    # Aquí devuelvo un mensaje JSON indicando que la imagen ha sido procesada
                return HttpResponse(image_data, content_type="image/png")
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

    def process_bw_image_google(request):
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

    @ csrf_exempt
    def process_bw_image_pc(request):
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
