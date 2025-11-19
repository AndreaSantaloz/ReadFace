# 🎭 Detectores de Rostros - Explicación Extremadamente Detallada

## Autora
Andrea Santana López

---

# 🎭 **DETECTOR DE EMOCIONES - Explicación Paso a Paso**

## **1. Configuración Inicial y Importaciones**

```python
import cv2
from ultralytics import YOLO
```

**Análisis Detallado:**
- **`cv2` (OpenCV)**: Biblioteca de visión por computadora que proporciona:
  - Captura de video en tiempo real
  - Procesamiento de imágenes
  - Dibujo de gráficos (rectángulos, texto)
  - Manejo de ventanas y eventos de teclado

- **`YOLO` de Ultralytics**: Implementación moderna del algoritmo YOLO para:
  - Detección de objetos en tiempo real
  - Alto rendimiento y precisión
  - Fácil integración y uso

## **2. Sistema de Carga de Emoticonos - Explicación Profunda**

```python
EMOTICONS = {}

def load_icon(path, force_bgr=False):
    # IMREAD_UNCHANGED: Preserva el canal alfa (transparencia) si existe
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    
    # Verificación de carga exitosa
    if img is None:
        print(f"Advertencia: No se pudo cargar el archivo: {path}")
        return None
    
    # Conversión forzada a BGR para imágenes sin transparencia
    if force_bgr and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img
```

**Conceptos Clave Explicados:**

### **`cv2.IMREAD_UNCHANGED`**
- **Propósito**: Carga la imagen exactamente como está almacenada
- **Canales de color**:
  - **BGR**: 3 canales (Blue, Green, Red) - imágenes normales
  - **BGRA**: 4 canales (Blue, Green, Red, Alpha) - imágenes con transparencia
- **Canal Alpha**: Controla la opacidad (0 = transparente, 255 = opaco)

### **Proceso de Carga Individual:**
```python
# Carga con transparencia preservada
emoticono_feliz = load_icon('feliz.png')
# FORCE_BGR = True convierte BGRA → BGR (elimina transparencia)
emoticono_neutral = load_icon('neutral.jpg', force_bgr=True)
```

## **3. Diccionario de Emoticonos y Colores**

```python
# Mapeo de emociones → imágenes de emoticonos
EMOTICONS = {
    "happy": emoticono_feliz,
    "sad": emoticono_triste, 
    "anger": emoticono_enfadado,
    # ... más emociones
}

# Sistema de colores para bounding boxes
color_emotions = {
    "happy": (0, 255, 0),      # VERDE en BGR
    "sad": (255, 0, 0),        # AZUL en BGR  
    "anger": (0, 0, 255),      # ROJO en BGR
    "surprise": (255, 255, 0), # CIAN en BGR
    # ... más colores
}
```

**Explicación del Formato BGR:**
- OpenCV usa **BGR** en lugar de RGB
- `(0, 255, 0)` = Verde puro (0 azul, 255 verde, 0 rojo)
- `(255, 0, 0)` = Azul puro (255 azul, 0 verde, 0 rojo)

## **4. Inicialización de Cámara y Modelo**

```python
capture_video = cv2.VideoCapture(0)
model = YOLO("runs/detect/train2/weights/best.pt")
```

**Desglose Técnico:**

### **`cv2.VideoCapture(0)`**
- **Parámetro `0`**: Índice de la cámara (0 = cámara principal)
- **Alternativas**: 
  - `1` = cámara secundaria
  - `"ruta/video.mp4"` = archivo de video

### **`YOLO("ruta/modelo.pt")`**
- **`.pt`**: Archivo de pesos del modelo entrenado
- **`best.pt`**: Mejores pesos del entrenamiento

## **5. Bucle Principal - Procesamiento Frame por Frame**

```python
while True:
    ret, frame = capture_video.read()
    if not ret:
        break
```

**Variables del Frame:**
- **`ret`**: Boolean que indica si la lectura fue exitosa
- **`frame`**: Matriz NumPy que representa la imagen (alto × ancho × canales)

## **6. Proceso de Detección de Emociones**

```python
results = model(frame, conf=0.5, verbose=False)

for result in results:
    boxes = result.boxes
    for box in boxes:
        # Extraer coordenadas del bounding box
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # Obtener confianza y clase
        confidence = box.conf[0].item()
        class_id = int(box.cls[0].item())
        label = model.names.get(class_id, "unknown")
```

**Análisis de Coordenadas:**
- **`box.xyxy[0]`**: Coordenadas en formato [x1, y1, x2, y2]
  - `(x1, y1)`: Esquina superior izquierda
  - `(x2, y2)`: Esquina inferior derecha
- **`confidence`**: Probabilidad de acierto (0.0 a 1.0)
- **`class_id`**: ID numérico de la clase detectada

## **7. Dibujo del Bounding Box y Etiqueta**

```python
# Dibujar rectángulo alrededor del rostro
cv2.rectangle(frame, (x1, y1), (x2, y2), color_emotion, 2)

# Añadir texto con emoción y confianza
cv2.putText(frame, f"{label} {confidence:.2f}", 
           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_emotion, 2)
```

**Parámetros de `cv2.rectangle`:**
- `frame`: Imagen donde dibujar
- `(x1, y1)`, `(x2, y2)`: Esquinas del rectángulo
- `color_emotion`: Color en formato BGR
- `2`: Grosor de la línea en píxeles

## **8. Superposición de Emoticonos - Algoritmo Complejo**

```python
if label in EMOTICONS and EMOTICONS[label] is not None:
    emotion_icon = EMOTICONS[label]
    face_width = x2 - x1
    ICON_SIZE = int(face_width * 0.5)  # 50% del ancho de la cara
```

### **Cálculo de Tamaño y Posición:**

```python
# Redimensionar emoticono proporcionalmente al rostro
icon_resized = cv2.resize(emotion_icon, (ICON_SIZE, ICON_SIZE))

# Posicionar encima de la cabeza
y_pos = y1 - ICON_SIZE  # Arriba del bounding box
x_pos = x1 + (face_width - ICON_SIZE) // 2  # Centrado horizontal
```

## **9. Manejo Avanzado de Transparencia (Alpha Blending)**

```python
if icon_resized.shape[2] == 4:  # Imagen con canal alfa
    # Separar canales de color y alpha
    b, g, r, a = cv2.split(icon_resized)
    alpha = a / 255.0  # Normalizar a 0.0-1.0
    inv_alpha = 1.0 - alpha
    
    # Recombinar sin canal alpha
    icon_bgr = cv2.merge([b, g, r])
    
    # Aplicar blending para cada canal de color
    for c in range(0, 3):
        frame[y_pos:y_pos+ICON_SIZE, x_pos:x_pos+ICON_SIZE, c] = \
            (alpha * icon_bgr[:, :, c]) + \
            (inv_alpha * frame[y_pos:y_pos+ICON_SIZE, x_pos:x_pos+ICON_SIZE, c])
else:
    # Sin transparencia - reemplazo directo
    frame[y_pos:y_pos+ICON_SIZE, x_pos:x_pos+ICON_SIZE] = icon_resized
```

**Fórmula de Alpha Blending:**
```
píxel_final = (alpha * píxel_emoticono) + ((1 - alpha) * píxel_fondo)
```

## **10. Manejo de Errores y Casos Extremos**

```python
# Verificar que el emoticono no se salga de los bordes
y_start = max(0, y_pos)
x_start = max(0, x_pos)
y_end = min(frame.shape[0], y_pos + ICON_SIZE)
x_end = min(frame.shape[1], x_pos + ICON_SIZE)

# Ajustar dimensiones si es necesario
if y_end - y_start != ICON_SIZE or x_end - x_start != ICON_SIZE:
    icon_resized = icon_resized[0:y_end-y_start, 0:x_end-x_start]
```

---

# 🤪 **FILTRO TIKTOK - Explicación Profunda**

## **1. Arquitectura del Sistema**

```python
import cv2
import dlib
import numpy as np
import imutils
import os
```

**Propósito de Cada Librería:**
- **`dlib`**: Detección de landmarks faciales (68 puntos)
- **`numpy`**: Operaciones matemáticas con matrices
- **`imutils`**: Utilidades para procesamiento de imágenes
- **`os`**: Manejo de rutas de archivos

## **2. Configuración del Predictor Facial**

```python
det = dlib.get_frontal_face_detector()
pred = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
```

**Componentes de dlib:**
- **`detector`**: Encuentra rostros en la imagen
- **`predictor`**: Localiza 68 puntos específicos en cada rostro

## **3. Landmarks Faciales - Los 68 Puntos**

Los 68 puntos se dividen en:
- **Puntos 0-16**: Contorno de la cara
- **Puntos 17-21**: Ceja derecha
- **Puntos 22-26**: Ceja izquierda  
- **Puntos 27-35**: Nariz
- **Puntos 36-41**: Ojo derecho
- **Puntos 42-47**: Ojo izquierdo
- **Puntos 48-67**: Boca

## **4. Función `mask_feature` - Análisis Exhaustivo**

```python
def mask_feature(image, pts, x_crop, y_crop, w_crop, h_crop, blur_kernel=11):
    # 1. Crear máscara binaria del rasgo
    feature_mask = np.zeros((h_crop, w_crop), dtype=np.uint8)
    
    # 2. Convertir coordenadas absolutas a relativas al recorte
    pts_rel = pts.copy()
    pts_rel[:, 0] -= x_crop  # Ajustar coordenada X
    pts_rel[:, 1] -= y_crop  # Ajustar coordenada Y
    
    # 3. Rellenar polígono convexo con los puntos del rasgo
    cv2.fillConvexPoly(feature_mask, pts_rel, 255)
    
    # 4. Suavizar bordes con filtro Gaussiano
    feature_mask_soft = cv2.GaussianBlur(feature_mask, (blur_kernel, blur_kernel), 0)
    
    # 5. Extraer región de interés de la imagen original
    crop = image[y_crop:y_crop + h_crop, x_crop:x_crop + w_crop]
    
    return crop, feature_mask_soft
```

## **5. Proceso de Desenfoque Facial Selectivo**

```python
# Crear máscara elíptica para la cara
face_mask = np.zeros(frame.shape[:2], dtype="uint8")
cv2.ellipse(face_mask, (x + w//2, y + h//2), (w//2, int(h*0.55)), 0, 0, 360, 255, -1)

# Aplicar desenfoque Gaussiano
blurred = cv2.GaussianBlur(frame, BLUR_FACE, 0)

# Combinar cara desenfocada con fondo nítido
output = cv2.add(
    cv2.bitwise_and(blurred, mask3),      # Cara desenfocada
    cv2.bitwise_and(frame, inv)           # Fondo nítido
)
```

**Lógica Bitwise:**
- **`bitwise_and(src1, src2)`**: Operación AND píxel a píxel
- **Máscara**: Donde es blanca (255), pasa la imagen; donde es negra (0), la bloquea

## **6. Procesamiento de Rasgos Individuales**

### **Extracción de Coordenadas:**
```python
features = {
    "left_eye": pts[36:42],    # 6 puntos del ojo izquierdo
    "right_eye": pts[42:48],   # 6 puntos del ojo derecho  
    "mouth": pts[48:68]        # 20 puntos de la boca
}
```

### **Cálculo de Región de Recorte:**
```python
x_min, y_min = np.min(fpts, axis=0)  # Esquina superior izquierda
x_max, y_max = np.max(fpts, axis=0)  # Esquina inferior derecha

# Añadir margen de seguridad (CROP píxeles)
xc = max(0, x_min - CROP)
yc = max(0, y_min - CROP)
wc = (x_max - x_min) + CROP * 2
hc = (y_max - y_min) + CROP * 2
```

## **7. Transformación y Reubicación de Rasgos**

### **Escalado Diferencial:**
```python
# Ojos: escalado con compresión horizontal
sw = SCALE * (HCOMP if name != "mouth" else 1)
sh = SCALE

nw = int(feat.shape[1] * sw)  # Nuevo ancho
nh = int(feat.shape[0] * sh)  # Nuevo alto
```

### **Posicionamiento Inteligente:**
```python
if name == "left_eye":
    nx = cx - nw - eye_gap      # Izquierda del centro
    ny = base_y - nh // 2       # Arriba de la línea base
elif name == "right_eye":
    nx = cx + eye_gap           # Derecha del centro  
    ny = base_y - nh // 2
elif name == "mouth":
    nx = cx - nw // 2           # Centrado horizontal
    ny = base_y + int(h * 0.05) # Ligeramente abajo
```

## **8. Algoritmo de Mezcla (Blending) con Máscara Suavizada**

```python
# 1. Extraer región de destino
region = output[y1:y2, x1:x2]

# 2. Ajustar tamaño si es necesario
feat_res = feat_res[:region.shape[0], :region.shape[1]]
mask_res = mask_res[:region.shape[0], :region.shape[1]]

# 3. Aplicar fórmula de blending
blended = (region * (1 - mask_res) + feat_res * mask_res).astype(np.uint8)

# 4. Reemplazar región en imagen de salida
output[y1:y2, x1:x2] = blended
```

**Fórmula Matemática del Blending:**
```
píxel_resultante = (píxel_fondo × (1 - máscara)) + (píxel_rasgo × máscara)
```

Donde:
- **máscara**: Valor entre 0.0 (transparente) y 1.0 (opaco)
- **píxel_fondo**: Imagen desenfocada
- **píxel_rasgo**: Rasgo facial procesado

## **9. Manejo de Bordes y Casos Límite**

```python
# Verificar que no nos salimos de los límites de la imagen
y1 = max(0, ny)
y2 = min(output.shape[0], ny + nh)
x1 = max(0, nx)  
x2 = min(output.shape[1], nx + nw)

# Si hay desbordamiento, recortar el rasgo
hp, wp = region.shape[:2]  # Dimensiones disponibles
feat_res = feat_res[:hp, :wp]  # Recortar rasgo
mask_res = mask_res[:hp, :wp]  # Recortar máscara
```

---





## **Detección Facial con dlib**

### **Proceso de Landmark Detection:**
1. **Detección de rostros** → Rectángulos delimitadores
2. **Predicción de landmarks** → 68 puntos por rostro
3. **Procesamiento individual** → Cada rasgo por separado

### **Ventajas de dlib:**
- Precisión en landmarks faciales
- Robustez frente a variaciones de iluminación
- Buen rendimiento en tiempo real

---

