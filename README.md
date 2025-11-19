# Detectores de rostros

## Autora
Andrea Santana López

## ¿Qué se ha desarrollado?
En primer lugar se ha planteado realizar un detector de aspectos biométricos de los rostros de personas, en el caso de este proyecto, se ha escogido como aspecto biométrico las emociones de las personas que son las siguientes: enfado,desagrado,miedo,felicidad,tristeza,sorpresa y neutro.

En segundo lugar se ha planteado un filtro de caras emitando a uno ya hecho en ticktock donde subimos los ojos y boca en la frente.

## ¿Cómo se ha desarrollado el detector de aspectos biométricos de emociones?
Para empezar se descarga un dataset de Roboflow de un modelo relacionado con nuestro detector de emociones que se entrenará donde  podra acceder a traves de este enlace ["enlace a roboflow"]("").
Para traer el modelo se realiza el siguiente código
```python
from roboflow import Roboflow
rf = Roboflow(api_key="API_Key_Pones_Aquí")
project = rf.workspace("akin-sn8wm").project("face-emotions")
version = project.version(2)
dataset = version.download("yolov11")
                
```

## ¿Cómo se ha desarrollado el filtro de cara tipico de ticktock?