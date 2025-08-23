PyroSDIS y FASDD: Guía de Descarga de
Datasets 🔥
Los datasets PyroSDIS y FASDD son colecciones recientes de imágenes para detección
de humo/fuego. A continuación se explican métodos para obtener cada uno, asegurando su
compatibilidad tanto con modelos YOLOv8 (detección por bounding boxes) como con
arquitecturas CNN+LSTM (modelos secuenciales para análisis temporal). También se
detallan sus formatos de anotación y consideraciones de uso.

Dataset PyroSDIS (Pyronear/SDIS)
PyroSDIS es un conjunto de imágenes de cámaras fijas orientado a la detección temprana
de humo de incendios forestales. Fue desarrollado por la iniciativa Pyronear en
colaboración con bomberos de Francia (SDIS) y voluntarios, con más de 33,600 imágenes
(unas 28,100 con humo) anotadas con cajas delimitadoras (bounding boxes) de
humohuggingface.co. El dataset está alojado en Hugging Face y ya viene formateado para
ser compatible con la librería Ultralytics YOLO (formato YOLO
estándar)huggingface.co, lo que facilita su uso en modelos YOLOv8 sin conversiones
adicionales.

Método de descarga (Hugging Face): Para obtener PyroSDIS, la forma recomendada es
usar la API de Hugging Face Datasets vía Python, ya que el dataset se publica en ese hub.
Pasos básicos:

Instalar dependencias: asegúrate de tener datasets de HuggingFace instalado (y
opcionalmente ultralytics si planeas entrenar de inmediato). Por ejemplo: pip
install datasets ultralyticshuggingface.co.
Cargar el dataset: utiliza load_dataset con el identificador del repositorio. Por
ejemplo en Python:
from datasets import load_dataset
dataset = load_dataset("pyronear/pyro-sdis")
Esto descargará los datos (aprox. 3.3 GB de imágenes) desde Hugging
Facehuggingface.co. El objeto dataset contendrá las divisiones predefinidas (e.g.
train y val).
Exportar a archivos locales: El dataset en Hugging Face está almacenado en un
formato interno (parquet) por eficiencia. Para usarlo con YOLOv8, conviene extraer
las imágenes y etiquetas a directorios. La página oficial provee un snippet de
ejemplo que recorre cada entrada y guarda la imagen y su etiqueta YOLO en
carpetas locales correspondienteshuggingface.cohuggingface.co. En resumen, debes
guardar las imágenes en images/train/ y images/val/, y los archivos de texto de
anotaciones YOLO en labels/train/ y
labels/val/huggingface.cohuggingface.co. Cada archivo de etiqueta .txt contiene
la línea con la clase (p.ej., 0 para humo) y los cuatro valores normalizados de
bounding box (x_centro, y_centro, ancho, alto), tal como es estándar en
YOLOhuggingface.cohuggingface.co. Nota: PyroSDIS es monoclase (solo humo),
por lo que Ultralytics recomienda habilitar single_cls=True al
entrenarhuggingface.co.
Descargar el archivo de configuración: La página de Hugging Face ofrece un
data.yaml listo para YOLOv8 con la estructura y clases. Puedes descargarlo
programáticamente usando
huggingface_hub.hf_hub_download("pyronear/pyro-sdis", "data.yaml",
repo_type="dataset")huggingface.co, o manualmente desde la sección Files del
dataset. Este YAML define rutas de entrenamiento/validación e indica que es una
sola clase de detección.
Verificar estructura: Tras exportar, tu carpeta debería lucir como:
pyro-sdis/
├── images/
│ ├── train/... (.jpg)
│ └── val/... (.jpg)
└── labels/
├── train/... (.txt)
└── val/... (.txt)
Cada imagen de entrenamiento tiene un .txt con la misma base de nombre en
labels/train (y análogo para val)huggingface.co.
Con lo anterior, PyroSDIS queda listo para entrenar un modelo YOLOv8 de Ultralytics, por
ejemplo ejecutando: yolo detect train data=data.yaml model=yolov8n.pt
epochs=... según las instrucciones oficialeshuggingface.co.

Formato y uso con YOLOv8: Las anotaciones vienen en formato YOLO (coordenadas
normalizadas) y el dataset fue diseñado específicamente para YOLO, así que no requiere
conversiónhuggingface.co. Es compatible directamente con la librería Ultralytics. Además,
la gran cantidad de ejemplos de “no humo” vs “humo” (imágenes negativas y positivas)
permiten entrenar detectores robustos. Uso en CNN+LSTM: PyroSDIS consiste en
imágenes estáticas etiquetadas individualmente, no secuencias de video. No obstante,
muchas provienen de cámaras temporales con timestamps. Si se desea usar en un modelo
secuencial (por ejemplo, evaluar detección de humo en series de imágenes), se podría
agrupar imágenes por cámara y orden temporal usando los metadatos (cada archivo incluye
el nombre de cámara y fechahuggingface.co). En su versión actual, PyroSDIS es
principalmente para detección en imágenes individuales, pero podría complementarse con
la parte de videos de PyroNear-2024 para secuencias más largas.

Dataset FASDD (Flame and Smoke Detec8on Dataset)
FASDD es un extenso dataset público (100k+ imágenes) enfocado en la detección de
flamas (fuego) y humo en múltiples escenarios. Incluye fotografías heterogéneas: desde
incendios forestales y urbanos (cámaras terrestres, drones) hasta imágenes satelitales,
abarcando condiciones diurnas/nocturnas, interiores/exteriores, etc.. Cada imagen puede
contener anotaciones de tipo bounding box para dos categorías : fire (llama/fuego

visible) y smoke (humo). Muchas imágenes tienen múltiples boxes (p. ej. varios focos de
fuego o varias columnas de humo) o pueden no tener ninguna si son negativas (sin
incendio). No se proporcionan segmentaciones por píxel, solo cajas delimitadoras alrededor
de las regiones de interés.

FASDD está disponible de dos maneras: mediante su DOI oficial (Science Data Bank de
China) y a través de una copia en Kaggle. A continuación, se detallan ambos métodos:

A) Descarga desde el DOI oficial (Science Data Bank)
Los autores han publicado FASDD en el repositorio nacional Science Data Bank con el
DOI 10.57760/sciencedb.j00104.0010 3 essd.copernicus.org. Para descargarlo:

Acceder al enlace DOI: Ingresa a la URL proporcionada (por ejemplo vía
doi.org/10.57760/sciencedb.j00104.00103). Esto redirige a la página del
dataset en Science Data Bank (que ofrece interfaz en inglés y chino). Allí debería
aparecer una opción de Download o Data.
Registro/Inicio de sesión: Es posible que debas crear una cuenta gratuita en
scidb.cn para poder descargar, dado el tamaño del dataset. La página en inglés
facilita la navegaciónmdpi.com. Sigue las instrucciones para iniciar la descarga del
paquete de datos.
Contenido de la descarga: El dataset suele venir empaquetado en un archivo ZIP
grande que incluye las imágenes y varias subcarpetas de anotaciones en diferentes
formatos estándar. Por ejemplo, al descomprimir podrías ver una estructura como:
o JPEGImages/ – contiene todas las imágenes en formato .jpg.
o Annotations_COCO/ – anotaciones en un único archivo JSON estilo COCO
(listas para usar con detectores que aceptan COCO; contiene categorías
"fire" y "smoke" y todas las bounding boxes).
o Annotations_PASCAL/ – anotaciones duplicadas en XML estilo Pascal
VOC (un .xml por imagen).
o Annotations_YOLO/ – anotaciones en texto estilo YOLO (un .txt por
imagen, con líneas <x_centro> <y_centro>
normalizados).
o Posiblemente archivos de división (train.txt, val.txt) con listas de
nombres de imagen para cada split (según reportes de algunos usuarios).
En otras palabras, FASDD provee las anotaciones en cuatro formatos para
conveniencia inmediata. Esto significa que no necesitas convertir manualmente
las etiquetas: si planeas usar YOLOv8, puedes tomar directamente las anotaciones
de Annotations_YOLO/; si prefieres COCO, usa el JSON de Annotations_COCO/,
etc.
Estructura y uso: Organiza las imágenes y anotaciones según requieras. Por
ejemplo, para YOLOv8 podrías mover las imágenes a images/train/,
images/val/ y las .txt correspondientes a labels/train/, labels/val/,
utilizando los splits que los autores definieron (si fueron provistos). De hecho,
FASDD se distribuyó con un split predefinido train/val en formato COCO, pero
puedes ajustarlo. Verifica la consistencia: son alrededor de 100 mil imágenes en
total, con decenas de miles de bounding boxes combinadas.
Obstáculos potenciales: La descarga oficial puede ser pesada (~5–6 GB
comprimido) y algo lenta debido al servidor internacional. Además, la interfaz de
Science Data Bank, aunque tiene versión en inglés, puede ser menos familiar. Si
experimentas lentitud o dificultades, considera la alternativa Kaggle abajo. Tip: La
licencia de FASDD es Creative Commons Attribution 4.0 (CC BY 4.0) , lo que
permite redistribuciones como Kaggle, por lo que usar la copia alternativa no
infringe condiciones mientras cites la fuente original.
B) Descarga alternativa desde Kaggle
Como alternativa más accesible, existe una copia de FASDD en Kaggle Datasets. Un
usuario (yuulind) ha subido el dataset bajo el nombre “FASDD_CV COCO Split” , que
contiene las imágenes y anotaciones en formato COCOkaggle.com. Esta versión se centra
en el componente de visión por computador (imágenes RGB terrestres y de dron); de
hecho, incluye todas las categorías de fuego/humo, pero puede excluir datos satelitales
dependiendo de cómo se filtró (está orientada a COCO, es decir, probablemente todas las
imágenes relevantes para detección en cámara convencional).

Opción 1 – Descarga via navegador web:

Inicia sesión en tu cuenta de Kaggle (debes tener una, pues Kaggle requiere login
para descargas). Ve a la página del dataset FASDD_CV COCO Split. Allí
encontrarás una descripción y una sección de Data con archivos.
Puedes descargar manualmente el dataset completo haciendo clic en el botón
“Download All” (o descargar archivos individuales si estuvieran separados). Por lo
general, el autor ha provisto un solo archivo comprimido grande .zip (~5 GB) que
contiene las imágenes y uno o más archivos JSON de anotaciones.
Ten en cuenta que Kaggle puede mostrar un mensaje de advertencia de tamaño y
pedir confirmación. Acepta para iniciar la descarga. Guarda el archivo ZIP en tu
equipo.
Opción 2 – Descarga mediante Kaggle API (CLI):

Instala la herramienta CLI de Kaggle (pip install kaggle) y configura tu API
Token (descargando kaggle.json desde tu perfil de Kaggle y colocándolo en
~/.kaggle/). Esto te permitirá descargar datasets vía línea de comandos.
Ejecuta el comando Kaggle datasets para FASDD. Por ejemplo:
kaggle datasets download -d yuulind/fasdd-cv-coco
Esto iniciará la descarga del dataset completo en tu directorio actual (archivo ZIP).
La consola indicará progreso; el tamaño es de aproximadamente 5 GB.
Una vez completado, descomprime el .zip. Obtendrás típicamente una carpeta con
todas las imágenes (.jpg) y el archivo de anotaciones en COCO JSON
(posiblemente separado en COCO_train.json y COCO_val.json, según cómo se
haya preparado el split). Confirma que el número de imágenes coincida (~100k).
Sugerencia: dado el volumen, es recomendable verificar hashes o conteos para
asegurarse de la integridad de la descarga.
Uso y formato (FASDD): La versión Kaggle viene en formato COCO por defecto, con
dos clases (fire y smoke). Para utilizarla con YOLOv8, tienes dos caminos:

Opción 1: Convertir las anotaciones COCO a formato YOLO. Puedes emplear
scripts o notebooks (por ejemplo, usando roboflow o utilidades de ultralytics
que importan COCO). Esto generará .txt por imagen. Dado que los organizadores
de FASDD ya proveían YOLO en la fuente original, también podrías optar por bajar
esa subcarpeta desde el ZIP oficial para ahorrarte la conversión.
Opción 2: Aprovechar que Ultralytics YOLOv8 puede leer COCO: en tu archivo
data.yaml, puedes especificar la ruta del JSON de COCO para train/val (consulta
documentación de Ultralytics). Internamente, YOLOv8 puede mapearlo. No
obstante, si prefieres simplicidad, la conversión a YOLO txt con un script Python es
directa.
En cuanto a la compatibilidad con modelos secuenciales (CNN+LSTM) , FASDD no
está organizado en secuencias temporales sino como imágenes independientes
recopiladas de múltiples fuentes. No hay series de frames de un mismo evento como tal,
por lo que no es un dataset pensado para entrenamiento temporal directo. Si tu objetivo es
un modelo de video (por ejemplo, detectar humo a lo largo de varios frames consecutivos),
FASDD serviría más bien como datos complementarios para pre-entrenar un detector
estático de humo/fuego. Para secuencias, datasets como FIgLib o PyroNear (que contienen
frames consecutivos) son más adecuados. FASDD sí garantiza un amplio rango de
apariencias de humo y fuego, útil para robustez del modelo de visión.

Resumen: FASDD se puede obtener oficialmente via Science Data Bank (formato
completo con COCO, VOC, YOLO ya incluidos) o más fácilmente via Kaggle. Incluye
bounding boxes de “fire” y “smoke” en todas las imágenes; no requiere anotación
adicional para YOLOv8, solo formateo según el método elegido. Su licencia CC-BY 4.
permite su uso libre con atribución, por lo que recuerda citar a Wang et al. (2022) según la
referencia proporcionada por los autores al publicar cualquier trabajo derivado.

Posibles contratiempos: La descarga desde el DOI puede requerir registro y paciencia,
mientras que Kaggle requiere tener cuenta (y su API token para la vía programática). Por
fortuna, el dataset no es excesivamente pesado (gracias a compresión de imágenes) para su
tamaño en número de archivos. Tras conseguirlo, estarás listo para entrenar modelos de
detección de incendios con YOLOv8 (dos clases) o analizar imágenes con redes CNN; solo
recuerda que para arquitecturas con LSTM u otro componente temporal necesitarás
secuencias, las cuales este dataset no provee de origen.
