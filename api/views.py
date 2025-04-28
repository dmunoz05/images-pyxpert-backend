from sklearn.model_selection import StratifiedShuffleSplit
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse, HttpResponse
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from django.core.files.base import ContentFile
from scipy.io.wavfile import read, write
from rest_framework import viewsets
from io import BytesIO, StringIO
import matplotlib.pyplot as plt
import IPython.display as ipd
from scipy.stats import skew
from decouple import config
from pathlib import Path
import soundfile as sf
import librosa.display
import urllib.request
from PIL import Image
from glob import glob
import numpy as np
import librosa
import base64
import qrcode
import joblib
import pickle
import json
import pulp
import cv2
import os

class ProcessOptimization(viewsets.ModelViewSet):
    @csrf_exempt
    def solve_integer_programming(request):
        if request.method == 'POST':
            try:
                body_unicode = request.body.decode('utf-8')
                data = json.loads(body_unicode)

                problem_type = data.get('problem_type', 'max')
                num_vars = data.get('num_vars', 0)
                objective_coeffs = data.get('objective_coeffs', [])
                constraints = data.get('constraints', [])

                if problem_type == 'max':
                    problem = pulp.LpProblem("Problema_Programacion_Entera", pulp.LpMaximize)
                else:
                    problem = pulp.LpProblem("Problema_Programacion_Entera", pulp.LpMinimize)

                # Crear variables enteras
                variables = [pulp.LpVariable(f"x{i+1}", lowBound=0, cat=pulp.LpInteger) for i in range(num_vars)]

                # Función objetivo
                problem += pulp.lpSum([objective_coeffs[i] * variables[i] for i in range(num_vars)])

                # Restricciones
                for con in constraints:
                    expr = pulp.lpSum([con['coeffs'][i] * variables[i] for i in range(num_vars)])
                    if con['relation'] == '<=':
                        problem += expr <= con['rhs']
                    elif con['relation'] == '=':
                        problem += expr == con['rhs']
                    elif con['relation'] == '>=':
                        problem += expr >= con['rhs']

                # Resolver
                solver = pulp.PULP_CBC_CMD(msg=False)
                problem.solve(solver)

                # Resultado
                response_data = {
                    "status": pulp.LpStatus[problem.status],
                    "optimal_value": pulp.value(problem.objective),
                    "variables": [
                        {"name": v.name, "value": int(v.varValue) if v.varValue is not None else None}
                        for v in variables
                    ]
                }

                return JsonResponse(response_data)

            except Exception as e:
                return JsonResponse({'status': 'error', 'message': f'Error al resolver la programación entera: {str(e)}'})


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


class ProcessAudio(viewsets.ModelViewSet):
    @csrf_exempt
    def process_model_listening(request):
        if request.method == 'POST':
            try:
                # Leer el blob del audio del cuerpo de la solicitud
                body_unicode = request.body.decode('utf-8')
                body_data = json.loads(body_unicode)
                audio_base64 = body_data.get('audio')

                if audio_base64:
                    # Decodificar el audio base64
                    decode_split = audio_base64.split(',')[1]
                    decode_audio = base64.b64decode(decode_split)
                    # ba = ' '.join(format(x, 'b') for x in bytearray(decode_audio))
                    audio_data = np.frombuffer(decode_audio, dtype=np.int16)
                    # vector_bytes = decode_audio.tobytes()
                    # vector_bytes_str = str(decode_audio)
                    # vector_bytes_str_enc = vector_bytes_str.encode()
                    # bytes_np_dec = vector_bytes_str_enc.decode('unicode-escape').encode('ISO-8859-1')[2:-1]
                    # audio_data = np.frombuffer(bytes_np_dec, dtype=np.int16)


                    # Probar modelo
                    Total=1;
                    Muestreo=5000;
                    Total_caracteristicas = 5
                    X_new = np.zeros((Total,Total_caracteristicas))
                    senal = np.resize(audio_data, Muestreo)

                    #Extracción de características
                    Varianza = np.var(senal) #1 dato
                    Desviacion=np.std(senal) #1 dato
                    rms_amplitude = np.sqrt(np.mean(np.square(senal))) #1 dato
                    zero_crossings = np.where(np.diff(np.sign(senal)))[0]
                    zcr = len(zero_crossings) #1 dato
                    skewness = skew(senal) #1 dato

                    X_new[0,0] = Varianza
                    X_new[0,1] = Desviacion
                    X_new[0,2] = rms_amplitude
                    X_new[0,3] = zcr
                    X_new[0,4] = skewness
                    # X_new[i,5] = Y[i]

                    # Normalizar los datos
                    scaler = StandardScaler()
                    X_new_normalized = scaler.fit_transform(X_new)
                    X_new_normalized = scaler.transform(X_new)
                    # X_new_normalized = scaler.transform(X_new)
                    # X_new_normalized = scaler.transform(X_new)

                    # Obtener el directorio de trabajo actual
                    cwd = os.getcwd()
                    # current_directory = os.path.join(cwd, 'models', 'Modelo_listening.pkl')
                    # current_directory = 'Modelo_listening.pkl'

                    # Obtener el directorio de trabajo actual
                    cwd = os.getcwd()
                    current_directory = os.path.join(cwd, 'models', 'Modelo_listening.pkl')

                    # if ('Repositorios' in current_directory):
                    #     current_directory = 'models/Modelo_listening.pkl'
                    # else:
                    #     current_directory = current_directory

                    # Cargar archivos desde el sistema local
                    with open(current_directory, 'rb') as f:
                        modelo_entrenado = pickle.load(f) # deserialize using load()with open(current_directory, 'wb') as f:
                        # modelo_entrenado = joblib.load(f)

                    # _x_fit = scaler.fit(_x_new)

                    # x_fit = scaler.fit_transform(_x_new)
                    # _x_new_normalized = scaler.transform(_x_new)
                    # Calcular el valor máximo del array
                    max_value = X_new.max()
                    min_value = X_new.min()

                    # if np.isnan(X_new.max) & np.isnan(X_new.min):
                    if np.isnan(max_value) & np.isnan(min_value):
                        response = 'No se pudo capturar correctamente la voz, intenta de nuevo'
                    else:
                        if modelo_entrenado.predict(X_new) == 0:
                            response = 'Es un joven'
                        if modelo_entrenado.predict(X_new) == 1:
                            response = 'Es un niño'
                        if modelo_entrenado.predict(X_new) == 2:
                            response = 'Es un anciano'


                    # Leer archivo y guardar el audio
                    # audio_path = os.path.join(
                    #     os.getcwd(), 'models', 'audio.wav')

                    # audio_path2 = os.path.join(
                    #     os.getcwd(), 'models', 'audio2.wav')

                    # Editar audio existente
                    # Load audio file.
                    # rate, s = read(audio_path)
                    # s = audio_data.astype(np.double) / 32767  # Or 32768.

                    # # Add amplitude modulation.
                    # rate = 44100  # Sampling rate.
                    # m1 = 3  # Message sine 1 [Hz].
                    # m2 = 7  # Message sine 2 [Hz].
                    # n = len(s)
                    # t = np.linspace(0, n / rate, n, endpoint=False)
                    # m = np.sin(2 * np.pi * m1 * t)  # Message.
                    # m += np.sin(2 * np.pi * m2 * t)
                    # m /= 2
                    # s *= (1 + m)
                    # s /= 2
                    # # Save audio file (range of s is [-1, 1]).
                    # write(audio_path2, rate, np.round(s * 32767).astype(np.int16))

                    # Normalizar la señal de audio
                    # if np.max(np.abs(audio_data)) > 0:
                    #     audio_data = audio_data / np.max(np.abs(audio_data))  # Escalar la señal entre -1 y 1
                    #     # audio_data = (audio_data * 32767).astype(np.int16)  # Escalar la señal a 16 bits
                    #     audio_data = np.round(audio_data * 32767).astype(np.int16)
                    # audio_data = ((audio_data + audio_data.min()) *
                    #               (2 ** 15) / audio_data.ptp()).astype(np.int16)

                    # Tasa de muestreo (asumiendo 44100 Hz, puedes ajustar esto si es diferente)
                    # rate = 44100
                    # n = 44100     # Length [samples]
                    # f = 1000      # Frequency of the sine [Hz]
                    # t = np.linspace(0, n / rate, n, endpoint=False)
                    # s = np.sin(2 * np.pi * f * t)

                    # if np.max(np.abs(audio_data)) > 0:
                    #     audio_data = ((audio_data + audio_data.min()) * (2 ** 15) / audio_data.ptp()).astype(np.int16)

                    # with open(audio_path, 'wb') as wav_file:
                    #     wav_file.write(audio_data.tobytes())

                    # x, sr = librosa.load(audio_path, sr=None)

                    # Normalizar
                    # data_audio = audio_data / np.max(np.abs(audio_data))
                    # data_audio = np.round(data_audio * 32768).astype(np.int16)


                    # METODO 1
                    # rate, s = read(audio_path)
                    # rate = 44100
                    # s = audio_data.astype(np.double) / 32768  # Or 32768.
                    # duracion_audio = len(audio_data) / 44100

                    # # Add amplitude modulation.
                    # m1 = 3  # Message sine 1 [Hz].
                    # m2 = 7  # Message sine 2 [Hz].
                    # n = len(s)
                    # # t = np.linspace(0, n / audio_data.size, n, endpoint=False)
                    # t = np.linspace(0, n / rate, len(audio_data), endpoint=False)
                    # # m = np.sin(2 * m1 * t)  # Message.
                    # m = np.sin(2 * np.pi * m1 * t)  # Message.
                    # m += np.sin(2 * np.pi * m2 * t)
                    # m /= 2
                    # s *= (1 + m)
                    # s /= 2


                    # METODO 2
                    # Create waveform
                    # rate = 44100  # Sampling rate [samples/s]
                    # n = 44100     # Length [samples]
                    # f = 1000      # Frequency of the sine [Hz]
                    # t = np.linspace(0, n / rate, n, endpoint=False)
                    # left = np.sin(2 * np.pi * f * t)
                    # left *= np.linspace(1, 0, n)
                    # right = np.sin(2 * np.pi * f * t)
                    # right *= np.linspace(0, 1, n)
                    # s = np.vstack((left, right)).transpose()

                    # write(audio_path, rate,
                    #       np.round(s * 32767).astype(np.int16))


                    # METODO 3
                    # Parámetros de la señal de audio existente
                    # rate_audio = 44100  # Tasa de muestreo del audio existente [muestras/s].

                    # # Calcular la duración total del audio en segundos
                    # duracion_audio = len(audio_data) / rate_audio

                    # # Parámetros de la señal de la onda sinusoidal
                    # n = len(audio_data)  # Longitud de la señal de audio [muestras].
                    # f = 1000  # Frecuencia de la onda sinusoidal [Hz].
                    # t = np.linspace(0, duracion_audio, len(audio_data), endpoint=False)
                    # # t = np.linspace(0, n / rate_sine, n, endpoint=False)
                    # s_sine = np.sin(2 * np.pi * f * t)

                    # # Normalizar la señal de audio existente al rango [-1, 1].
                    # max_value_audio = np.max(np.abs(audio_data))
                    # audio_data_normalized = audio_data / max_value_audio if max_value_audio > 1 else audio_data

                    # # Normalizar la señal de la onda sinusoidal al rango [-1, 1].
                    # max_value_sine = np.max(np.abs(s_sine))
                    # s_sine_normalized = s_sine / max_value_sine if max_value_sine > 1 else s_sine

                    # # Combinar las dos señales de audio.
                    # s_combined = audio_data_normalized + s_sine_normalized

                    # # Normalizar la señal combinada al rango [-1, 1].
                    # max_value_combined = np.max(np.abs(s_combined))
                    # s_combined_normalized = s_combined / max_value_combined if max_value_combined > 1 else s_combined

                    # Guardar la señal de audio combinada en un nuevo archivo WAV.
                    # write(audio_path, rate_audio, np.round(s_combined_normalized * 32767).astype(np.int16))

                    # write(audio_path, 44100, s)

                    # x, sr = read(audio_path)

                    # X_New = np.resize(sr.astype(np.double) / 32768, 50000)  # Muestreo
                    # plt.plot(sr)
                    # plt.show()

                    return JsonResponse({'status': 'error', 'message': response})
                    # return HttpResponse({'status': 'success', 'message': 'Proceso exitoso'})
            except Exception as e:
                if e.args[0] == 'buffer size must be a multiple of element size':
                    return JsonResponse({'status': 'error', 'message': f'Error al procesar el audio, intente de nuevo'})
                return JsonResponse({'status': 'error', 'message': f'Error al procesar el audio: {str(e)}'})
        return JsonResponse({'status': 'error', 'message': 'Método no permitido'}, status=405)


class ProcessImages(viewsets.ModelViewSet):
    @csrf_exempt
    def process_qr_image_pc(request):
        if request.method == 'POST':
            try:
                # Leer la imagen del cuerpo de la solicitud
                body_unicode = request.body.decode('utf-8')
                body_data = json.loads(body_unicode)
                image_base64 = body_data.get('image')

                if image_base64:
                    # Decodificar la imagen base64
                    decode_image = base64.b64decode(image_base64)
                    _image = cv2.imdecode(np.frombuffer(
                        decode_image, dtype=np.uint8), -1)

                    # Obtener el directorio de trabajo actual
                    cwd = os.getcwd()
                    current_directory = os.path.join(
                        cwd, 'models', 'haarcascade_frontalface_default.xml')

                    if ('Repositorios' in current_directory):
                        Detector = cv2.CascadeClassifier(
                            'models/haarcascade_frontalface_default.xml')
                    else:
                        Detector = cv2.CascadeClassifier(current_directory)

                    I_gris = cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY)
                    Cara = Detector.detectMultiScale(I_gris, 1.1, 5)

                    for (x, y, w, h) in Cara:
                        Recorte = I_gris[y:y+h, x:x+w]
                        Recorte = cv2.resize(
                            Recorte, (48, 48), interpolation=cv2.INTER_AREA)

                    # Codificar la imagen recortada en base64
                    _, buffer = cv2.imencode('.jpg', Recorte)
                    base64_img = base64.b64encode(buffer).decode('utf-8')

                    # Generar el código QR
                    # img = qrcode.make(base64_img)
                    # image_data = base64.b64encode(img.decode("utf-8"))

                    # Generar el código QR
                    img = qrcode.make(base64_img)
                    type(img)
                    img.save("qr.png")

                    # Obtener el directorio de trabajo actual
                    cwd = os.getcwd()
                    current_directory = os.path.join(cwd, 'qr.png')

                    # Leer la imagen desde el archivo
                    with open(current_directory, "rb") as img_file:
                        # Codificar la imagen como base64
                        image_data = base64.b64encode(
                            img_file.read()).decode("utf-8")

                    # Decodificar la imagen base64
                    decode_image = base64.b64decode(image_data)
                    im_arr = np.frombuffer(decode_image, dtype=np.uint8)
                    Img = cv2.imdecode(im_arr, cv2.IMREAD_COLOR)

                    buffer = BytesIO()

                    # Convertir la imagen a BytesIO
                    Image.fromarray(Img).save(buffer, format='PNG')

                    # Obtener los datos de la imagen
                    image_qr = buffer.getvalue()

                    return HttpResponse(image_qr, content_type="image/png")
            except Exception as e:
                return JsonResponse({'status': 'error', 'message': 'Error al procesar la imagen: {}'.format(str(e))})

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
                        cwd, 'models', 'Modelo_pdc.pkl')

                    # Cargar archivos desde el sistema local
                    with open(current_directory, 'rb') as f:
                        # Con el KNN5
                        modelo_entrenado = pickle.load(f)
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
        if 'image_url' in request.GET:
            image_url = request.GET['image_url']
            try:
                # Leer imagen
                req = urllib.request.urlopen(image_url)
                arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
                _img = cv2.imdecode(arr, -1)

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
                    image = cv2.imdecode(im_arr, 1)

                    # Cambiar a gris
                    gray_image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
                    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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
