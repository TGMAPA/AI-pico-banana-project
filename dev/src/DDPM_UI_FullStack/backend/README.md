# DDPM Image Generator - Backend

Backend en Flask para desplegar modelos DDPM (Denoising Diffusion Probabilistic Models).

## Requisitos

- Python 3.10+
- pip

## Instalación

1. Navega a la carpeta del backend:
```bash
cd backend
```

2. Instala las dependencias:
```bash
pip install -r requirements.txt
```

## Configuración

### Agregar tu modelo DDPM

Coloca tu archivo de modelo `.pth` en la siguiente ubicación:
```
backend/model/model.pth
```

El servidor detectará automáticamente el modelo al iniciar.

**Nota:** Si no colocas un modelo, el servidor funcionará en modo desarrollo generando imágenes de muestra.

## Ejecución

Ejecuta el servidor Flask:
```bash
python app.py
```

El servidor estará disponible en: `http://localhost:5000`

## Endpoints

### GET `/`
Health check del servidor.

**Respuesta:**
```json
{
  "status": "online",
  "model_loaded": true,
  "device": "cuda",
  "model_path": "...",
  "image_size": [64, 64],
  "channels": 3
}
```

### POST `/generate`
Genera una imagen usando el modelo DDPM.

**Request Body (opcional):**
```json
{
  "image_size": [64, 64],
  "channels": 3,
  "num_steps": 50
}
```

**Respuesta:**
```json
{
  "image": "<base64_encoded_png>",
  "status": "success",
  "mode": "model"
}
```

### GET `/config`
Obtiene la configuración actual del servidor.

## Estructura

```
backend/
├── app.py                  # Servidor Flask principal
├── model/
│   ├── __init__.py
│   ├── ddpm_loader.py      # Carga del modelo
│   ├── ddpm_inference.py   # Lógica de inferencia
│   └── model.pth           # Tu modelo (colócalo aquí)
├── utils/
│   └── image_utils.py      # Utilidades de procesamiento de imágenes
├── static/                 # Archivos estáticos (opcional)
└── requirements.txt        # Dependencias Python
```

## Desarrollo

El servidor incluye comentarios detallados en cada archivo explicando la lógica implementada.

### Archivos principales:

- **app.py**: Servidor Flask con endpoints
- **ddpm_loader.py**: Carga y preparación del modelo
- **ddpm_inference.py**: Proceso de generación de imágenes
- **image_utils.py**: Conversión entre tensors, PIL y base64

## Notas

- El modelo debe estar en formato `.pth` de PyTorch
- Soporta imágenes RGB (3 canales) y grayscale (1 canal)
- El tamaño de imagen por defecto es 64x64
- CORS está configurado para permitir requests desde `http://localhost:5173`
