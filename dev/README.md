# DDPM Image Generator

Proyecto completo para desplegar modelos DDPM (Denoising Diffusion Probabilistic Models) con interfaz web.

## Estructura del Proyecto

```
UI-Test/
├── backend/                # Backend Flask (Python)
│   ├── app.py
│   ├── model/
│   │   ├── ddpm_loader.py
│   │   ├── ddpm_inference.py
│   │   └── model.pth       # Coloca tu modelo aquí
│   ├── utils/
│   │   └── image_utils.py
│   ├── requirements.txt
│   └── README.md
│
└── frontend/               # Frontend React
    ├── src/
    │   ├── App.jsx
    │   ├── main.jsx
    │   ├── styles.css
    │   └── components/
    │       └── ImageGenerator.jsx
    ├── package.json
    ├── vite.config.js
    ├── tailwind.config.js
    └── README.md
```

## Inicio Rápido

### 1. Backend (Flask)

```bash
cd backend
pip install -r requirements.txt
python app.py
```

El backend estará en: `http://localhost:5000`

### 2. Frontend (React)

En otra terminal:

```bash
cd frontend
npm install
npm run dev
```

El frontend estará en: `http://localhost:5173`

### 3. Usar tu modelo DDPM

Coloca tu archivo `.pth` en:
```
backend/model/model.pth
```

## Tecnologías

**Backend:**
- Python 3.10+
- Flask
- PyTorch
- PIL (Pillow)
- NumPy

**Frontend:**
- React 18
- Vite 5
- Tailwind CSS 3
- PostCSS

## API Endpoints

### `GET /`
Health check del servidor

### `POST /generate`
Genera una imagen usando DDPM

**Request:**
```json
{
  "image_size": [64, 64],
  "channels": 3,
  "num_steps": 50
}
```

**Response:**
```json
{
  "image": "<base64_png>",
  "status": "success",
  "mode": "model"
}
```

### `GET /config`
Configuración del servidor

## Uso

1. **Abrir la aplicación web** en `http://localhost:5173`
2. **Presionar "Generar Imagen"** para crear una nueva imagen
3. **Esperar** mientras el modelo genera la imagen
4. **Visualizar** la imagen generada
5. **Descargar** (opcional) la imagen en PNG


## Licencia

Este proyecto es de código abierto y está disponible para uso educativo.


# Guía de Inicio Rápido

## Inicio Rápido 

### Linux/Mac:
```bash
./start.sh
```

### Windows:
```cmd
start.bat
```

Estos scripts automáticamente:
- Verifican requisitos (Python, Node.js)
- Instalan dependencias
- Inician backend y frontend
- Abren los servicios en el navegador

---

## Inicio Manual

### Paso 1: Backend

```bash
cd backend

# Crear entorno virtual (opcional pero recomendado)
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate.bat

# Instalar dependencias
pip install -r requirements.txt

# Iniciar servidor
python app.py
```

Backend disponible en: http://localhost:5000

### Paso 2: Frontend

En otra terminal:

```bash
cd frontend

# Instalar dependencias
npm install

# Iniciar servidor de desarrollo
npm run dev
```

Frontend disponible en: http://localhost:5173

---

## Usar tu Modelo DDPM

### Opción 1: Modelo Simple
1. Coloca tu archivo `.pth` en: `backend/model/model.pth`
2. Reinicia el backend
3. ¡Listo! El servidor detectará automáticamente el modelo

### Opción 2: Modelo con Arquitectura Personalizada

Si tu modelo requiere una arquitectura específica:

1. Edita `backend/model/ddpm_loader.py`:
   - Define tu clase de modelo (ej: UNet)
   - Actualiza la función `load_model()` para instanciar tu arquitectura

2. Edita `backend/model/ddpm_inference.py`:
   - Actualiza `generate_image()` con tu lógica de inferencia
   - Implementa el proceso de denoising específico de tu modelo

3. Revisa `backend/EXAMPLE_CUSTOM_DDPM.py` para ejemplos detallados

---

## Verificar Instalación

### Test del Backend:

```bash
cd backend
python test_backend.py
```

Esto probará:
- Health check
- Generación de imágenes
- Configuración

### Test Manual:

**Backend:**
```bash
curl http://localhost:5000/
```

**Frontend:**
Abre http://localhost:5173 en tu navegador

---

## Uso de la Aplicación

1. **Abrir** http://localhost:5173 en tu navegador
2. **Presionar** el botón "✨ Generar Imagen"
3. **Esperar** mientras el modelo genera la imagen
4. **Ver** la imagen generada
5. **Descargar** (opcional) usando el botón "💾 Descargar"

---

## Estructura de Archivos

```
UI-Test/
├── backend/                    # Backend Flask
│   ├── app.py                 # Servidor principal
│   ├── model/
│   │   ├── ddpm_loader.py    # Carga del modelo
│   │   ├── ddpm_inference.py # Inferencia
│   │   └── model.pth         # Tu modelo (colócalo aquí)
│   ├── utils/
│   │   └── image_utils.py    # Utilidades de imagen
│   ├── requirements.txt       # Dependencias Python
│   └── test_backend.py       # Tests
│
├── frontend/                   # Frontend React
│   ├── src/
│   │   ├── App.jsx           # App principal
│   │   ├── main.jsx          # Entry point
│   │   └── components/
│   │       └── ImageGenerator.jsx  # Componente generador
│   ├── package.json          # Dependencias Node
│   └── vite.config.js        # Config Vite
│
├── start.sh                   # Script de inicio (Linux/Mac)
├── start.bat                  # Script de inicio (Windows)
└── README.md                  # Documentación principal
```


# Integration Guide - Dynamic Model Selection System

## System Overview

The DDPM Image Generator has been refactored to support dynamic model selection from a `SerializedModels/` directory. Users can now choose from any available model at runtime without code changes.

## Architecture

```
Frontend (React)          Backend (Flask)              File System
    │                         │                            │
    ├─ App.jsx               ├─ app.py                     │
    │                        │   ├─ /models (GET)          │
    │                        │   ├─ /generate (POST)       │
    │                        │   ├─ / (GET)                │
    │                        │   └─ /config (GET)          │
    │                        │                              │
    │                        ├─ model_manager.py            │
    │                        │   ├─ ModelManager            │
    │                        │   └─ ModelValidator          │
    │                        │                              │
    └─ ImageGenerator.jsx───┼─ ddpm_loader.py             ├─ SerializedModels/
       (Model Selector)      ├─ ddpm_inference.py          ├─ model1.pt
                             └─ image_utils.py            ├─ model2.pth
                                                           └─ model3.pt
```

## Setup Instructions

### 1. Prepare Models Directory

```bash
mkdir -p backend/model/SerializedModels
# Place your .pt or .pth files here
cp /path/to/your/models/*.pt backend/model/SerializedModels/
```

### 2. Backend Setup

**No additional installation required** - all dependencies already in `requirements.txt`

```bash
cd backend
pip install -r requirements.txt
python app.py
```

Backend will:
- Scan `SerializedModels/` directory
- List available models on startup
- Wait for requests on `http://localhost:5000`

### 3. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Frontend will:
- Start dev server on `http://localhost:5173`
- Fetch models from backend on load
- Display model selector

## Usage Flow

### User Perspective

1. **Load Application**
   - Open `http://localhost:5173` in browser
   - App fetches available models from backend
   - Model dropdown populates automatically
   - First model auto-selected

2. **Select Model**
   - Choose desired model from dropdown
   - Model name displayed in info panel

3. **Generate Image**
   - Click "Generate Image" button
   - Loading spinner appears
   - Backend processes request
   - Image appears when ready
   - Can download or generate again

### Behind the Scenes

**Frontend → Backend Request:**
```
POST http://localhost:5000/generate
Content-Type: application/json

{
  "model": "model1",
  "image_size": [64, 64],
  "channels": 3,
  "num_steps": 50
}
```

**Backend Processing:**
1. Validate model name (security check)
2. Check if model exists
3. Load model from disk (or cache)
4. Run inference with model
5. Convert output to base64
6. Send image back to frontend

**Backend → Frontend Response:**
```
{
  "status": "success",
  "image": "iVBORw0KGgoAAAANSUhEUgAA...",
  "mode": "model",
  "model_used": "model1",
  "image_size": [64, 64],
  "channels": 3
}
```

## API Reference

### GET /models

Fetch list of available models.

**Request:**
```bash
curl http://localhost:5000/models
```

**Response:**
```json
{
  "status": "success",
  "models": [
    {
      "name": "model1",
      "path": "/full/path/to/model1.pt",
      "extension": ".pt",
      "size_mb": 123.45
    },
    {
      "name": "model2",
      "path": "/full/path/to/model2.pth",
      "extension": ".pth",
      "size_mb": 234.56
    }
  ],
  "count": 2
}
```

**Status Codes:**
- `200`: Success
- `500`: Server error

---

### POST /generate

Generate image with selected model.

**Request:**
```json
{
  "model": "model1",
  "image_size": [64, 64],
  "channels": 3,
  "num_steps": 50
}
```

**Response (Success):**
```json
{
  "image": "<base64_encoded_png>",
  "status": "success",
  "mode": "model",
  "model_used": "model1",
  "image_size": [64, 64],
  "channels": 3
}
```

**Response (Error - Model not found):**
```json
{
  "status": "error",
  "message": "Model \"model1\" not found",
  "available_models": ["model2", "model3"]
}
```

**Status Codes:**
- `200`: Success
- `400`: Bad request (model not found, invalid name)
- `404`: Model file missing
- `500`: Server error

---

### GET /

Health check endpoint.

**Response:**
```json
{
  "status": "online",
  "models_available": 2,
  "device": "cuda",
  "cache_enabled": true,
  "image_size": [64, 64],
  "channels": 3
}
```

---

### GET /config

Get server configuration.

**Response:**
```json
{
  "device": "cuda",
  "cache_enabled": true,
  "cache_info": {
    "cached_models": 1,
    "model_names": ["model1"]
  },
  "models_directory": "/path/to/SerializedModels",
  "models_available": 2,
  "default_image_size": [64, 64],
  "default_channels": 3
}
```

## Configuration

### Backend Configuration

Edit `backend/app.py`:

```python
# Enable/disable model caching
ENABLE_MODEL_CACHE = True

# Default image generation size
IMAGE_SIZE = (64, 64)

# Default number of channels (3=RGB, 1=Grayscale)
CHANNELS = 3

# Models directory (relative to app.py)
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'model', 'SerializedModels')
```

### Frontend Configuration

Edit `frontend/src/components/ImageGenerator.jsx`:

```javascript
// Backend URL
const BACKEND_URL = 'http://localhost:5000'

// Generation parameters (can be modified)
image_size: [64, 64],
channels: 3,
num_steps: 50
```

## File Structure

```
UI-Test/
├── backend/
│   ├── app.py                              # Main server (UPDATED)
│   ├── model/
│   │   ├── model_manager.py               # NEW - Model management
│   │   ├── ddpm_loader.py                 # Model loading utilities
│   │   ├── ddpm_inference.py              # Image generation
│   │   ├── __init__.py
│   │   └── SerializedModels/              # NEW - Model storage
│   │       ├── model1.pt                  # Your models here
│   │       ├── model2.pth
│   │       └── ...
│   ├── utils/
│   │   └── image_utils.py                 # Image utilities
│   └── requirements.txt
│
└── frontend/
    ├── src/
    │   ├── App.jsx                        # Main app (UPDATED)
    │   ├── main.jsx
    │   ├── styles.css                     # Global styles (UPDATED)
    │   └── components/
    │       └── ImageGenerator.jsx         # Main component (UPDATED)
    ├── package.json
    ├── vite.config.js
    ├── tailwind.config.js
    └── postcss.config.js
```

## Model Format Requirements

### File Extensions
- `.pt` (PyTorch)
- `.pth` (PyTorch)

### Checkpoint Format

Models should be saved in one of these formats:

**Option 1: State Dict**
```python
torch.save(model.state_dict(), 'model.pt')
```

**Option 2: Full Model**
```python
torch.save(model, 'model.pt')
```

**Option 3: Checkpoint**
```python
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
}, 'model.pt')
```

All formats are supported automatically!


# 🔄 Flujo de Trabajo del Sistema DDPM

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ARQUITECTURA DEL SISTEMA                      │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────────┐         HTTP POST          ┌──────────────────┐
│                  │    /generate (JSON)         │                  │
│   FRONTEND       │ ──────────────────────────> │    BACKEND       │
│   React + Vite   │                             │   Flask + PyTorch│
│   Port: 5173     │ <────────────────────────── │   Port: 5000     │
│                  │    JSON (base64 image)      │                  │
└──────────────────┘                             └──────────────────┘
        │                                                  │
        │                                                  │
        ▼                                                  ▼
┌──────────────────┐                             ┌──────────────────┐
│  UI Components   │                             │  Model Loader    │
│  - ImageGenerator│                             │  - Load .pth     │
│  - Button        │                             │  - Detect GPU    │
│  - Loader        │                             │  - Prepare model │
│  - Image Display │                             └──────────────────┘
└──────────────────┘                                      │
                                                          ▼
                                                 ┌──────────────────┐
                                                 │  DDPM Inference  │
                                                 │  - Generate noise│
                                                 │  - Denoising loop│
                                                 │  - Return tensor │
                                                 └──────────────────┘
                                                          │
                                                          ▼
                                                 ┌──────────────────┐
                                                 │  Image Utils     │
                                                 │  - Tensor to PIL │
                                                 │  - PIL to base64 │
                                                 │  - Normalize     │
                                                 └──────────────────┘
```

---

## 📋 Flujo de Generación de Imagen

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PROCESO DE GENERACIÓN                             │
└─────────────────────────────────────────────────────────────────────┘

1. USUARIO
   │
   └─> Presiona "Generar Imagen"
       │
       ▼

2. FRONTEND (ImageGenerator.jsx)
   │
   ├─> setLoading(true)
   ├─> fetch('http://localhost:5000/generate', {...})
   └─> Muestra loader animado
       │
       ▼

3. BACKEND (app.py)
   │
   ├─> Recibe POST request
   ├─> Extrae parámetros (size, channels, steps)
   └─> Llama generate_image()
       │
       ▼

4. DDPM INFERENCE (ddpm_inference.py)
   │
   ├─> Inicializa ruido gaussiano
   ├─> Loop de denoising (num_steps veces)
   │   ├─> Predice ruido con modelo
   │   ├─> Aplica step de denoising
   │   └─> Agrega ruido si no es último step
   ├─> Normaliza resultado [0, 1]
   └─> Convierte tensor a PIL Image
       │
       ▼

5. IMAGE UTILS (image_utils.py)
   │
   ├─> tensor_to_pil()
   │   ├─> Mueve a CPU
   │   ├─> Clamp [0, 1]
   │   ├─> Convierte a numpy
   │   ├─> Transpone (C,H,W) → (H,W,C)
   │   └─> Escala a [0, 255]
   │
   └─> pil_to_base64()
       ├─> Guarda en BytesIO buffer
       ├─> Codifica a base64
       └─> Retorna string
       │
       ▼

6. BACKEND (app.py)
   │
   └─> Retorna JSON:
       {
         "image": "<base64_string>",
         "status": "success",
         "mode": "model"
       }
       │
       ▼

7. FRONTEND (ImageGenerator.jsx)
   │
   ├─> Recibe response
   ├─> setImageData(data.image)
   ├─> setLoading(false)
   └─> Renderiza: <img src={`data:image/png;base64,${imageData}`} />
       │
       ▼

8. USUARIO
   │
   └─> Ve la imagen generada ✅
```


# DDPM Image Generator - Frontend

Interfaz web en React para consumir el backend DDPM y visualizar imágenes generadas.

## Requisitos

- Node.js 16+
- npm o yarn

## Instalación

1. Navega a la carpeta del frontend:
```bash
cd frontend
```

2. Instala las dependencias:
```bash
npm install
```

## Ejecución

### Modo Desarrollo

```bash
npm run dev
```

La aplicación estará disponible en: `http://localhost:5173`

### Build para Producción

```bash
npm run build
```

### Preview del Build

```bash
npm run preview
```

## Estructura

```
frontend/
├── index.html              # HTML principal
├── package.json            # Dependencias y scripts
├── vite.config.js          # Configuración de Vite
├── tailwind.config.js      # Configuración de Tailwind
├── postcss.config.js       # Configuración de PostCSS
└── src/
    ├── main.jsx            # Punto de entrada React
    ├── App.jsx             # Componente principal
    ├── styles.css          # Estilos globales + Tailwind
    └── components/
        └── ImageGenerator.jsx  # Componente generador
```

## Configuración

### Backend URL

El frontend está configurado para conectarse a:
```javascript
const BACKEND_URL = 'http://localhost:5000/generate'
```

### Parámetros de Generación

Puedes personalizar los parámetros de generación en `ImageGenerator.jsx`:

```javascript
body: JSON.stringify({
  image_size: [64, 64],  // Tamaño de la imagen
  channels: 3,            // Canales (3=RGB, 1=Grayscale)
  num_steps: 50           // Pasos de denoising
})
```
