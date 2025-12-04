# DDPM Image Generator

A full-stack system for Denoising Diffusion Probabilistic Models (DDPM) featuring dynamic model selection, web-based inference, and comprehensive model training capabilities.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [System Architecture](#system-architecture)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Model Integration](#model-integration)
- [API Reference](#api-reference)
- [Image Generation Pipeline](#image-generation-pipeline)
- [Configuration](#configuration)
- [License](#license)
- [Contact](#contact)

---

## Project Overview

This project provides a complete infrastructure for developing, deploying, and serving DDPM-based image generation models. The system consists of three primary components:

1. **Model Development Environment**: PyTorch Lightning-based training infrastructure located in `src/model/`
2. **Backend Server**: Flask REST API with dynamic model loading and inference capabilities
3. **Frontend Interface**: React-based web application for interactive image generation

The system supports dynamic model selection at runtime, allowing users to switch between different trained models without code modifications.

---

## Features

- **Dynamic Model Selection**: Load and switch between multiple trained models at runtime
- **Model Validation and Caching**: Automatic model validation with intelligent caching for performance optimization
- **REST API**: Well-documented HTTP endpoints for model management and image generation
- **Interactive Web Interface**: Modern React-based UI with real-time generation feedback
- **Flexible Model Support**: Compatible with PyTorch `.pt` and `.pth` checkpoint formats
- **GPU Acceleration**: Automatic CUDA detection and utilization when available
- **Base64 Image Encoding**: Efficient image transfer via JSON responses
- **CORS-Enabled**: Configured for secure cross-origin requests between frontend and backend

---

## Technology Stack

### Backend
- **Python**: 3.10+
- **Flask**: Web framework for REST API
- **PyTorch**: Deep learning framework
- **PyTorch Lightning**: Model training and organization
- **PIL (Pillow)**: Image processing
- **NumPy**: Numerical operations
- **Flask-CORS**: Cross-origin resource sharing

### Frontend
- **React**: 18.2.0
- **Vite**: 5.0.8 (build tool and dev server)
- **Tailwind CSS**: 3.4.0 (utility-first styling)
- **PostCSS**: CSS transformations

### Model Development
- **PyTorch Lightning**: Training framework
- **Custom UNet Architecture**: Diffusion model backbone
- **DataModule Pipeline**: Data loading and preprocessing

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      SYSTEM ARCHITECTURE                        │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────┐      HTTP REST API       ┌──────────────────┐
│                  │                          │                  │
│   FRONTEND       │ ──────────────────────>  │    BACKEND       │
│   React + Vite   │                          │   Flask + PyTorch│
│   Port: 5173     │ <──────────────────────  │   Port: 5000     │
│                  │    JSON (base64 image)   │                  │
└──────────────────┘                          └──────────────────┘
                                                       │
                                                       │
                                              ┌────────▼────────┐
                                              │ Model Manager   │
                                              │ - Validation    │
                                              │ - Caching       │
                                              │ - Loading       │
                                              └────────┬────────┘
                                                       │
                                              ┌────────▼────────┐
                                              │ SerializedModels│
                                              │ - model1.pt     │
                                              │ - model2.pth    │
                                              └─────────────────┘
```

### Request Flow

1. User interacts with React frontend
2. Frontend sends POST request to `/generate` endpoint
3. Backend validates model selection and parameters
4. Model Manager loads requested model (from cache or disk)
5. DDPM inference engine generates image through denoising process
6. Image is converted to base64-encoded PNG
7. JSON response returned to frontend
8. Frontend decodes and displays image

---

## Directory Structure

```
dev/
├── data/                                    # Dataset storage and pipelines
│   ├── celeba/                             # CelebA dataset
│   ├── openimage_source_images/            # OpenImages dataset
│   └── exploration-scripts/                # Data exploration notebooks
│
├── src/
│   ├── config/                             # Global configuration
│   │   ├── config.py                       # Configuration settings
│   │   └── libraries.py                    # Shared imports
│   │
│   ├── model/                              # Model development codebase
│   │   ├── main.py                         # Training entry point
│   │   ├── PicoBanana.py                   # Main model definition
│   │   ├── resume_from_checkpoint.py       # Checkpoint resumption
│   │   ├── DataModule/                     # Data loading pipeline
│   │   ├── DiffusionFwd/                   # Forward diffusion process
│   │   ├── DiffusionReversed/              # Reverse diffusion (denoising)
│   │   ├── Unet/                           # UNet architecture
│   │   ├── LightningModule/                # PyTorch Lightning modules
│   │   ├── Evaluation/                     # Model evaluation utilities
│   │   ├── Metrics_Plots/                  # Training metrics and visualization
│   │   ├── ModelCheckpoints/               # Training checkpoints
│   │   ├── SerializedObjects/              # Serialized artifacts
│   │   └── TrainingLogs/                   # TensorBoard logs
│   │
│   └── DDPM_UI_FullStack/
│       ├── backend/
│       │   ├── app.py                      # Flask server entry point
│       │   ├── requirements.txt            # Python dependencies
│       │   ├── model/
│       │   │   ├── model_manager.py        # Model management system
│       │   │   ├── ddpm_loader.py          # Model loading utilities
│       │   │   ├── ddpm_inference.py       # Inference engine
│       │   │   └── SerializedModels/       # Production model storage
│       │   │       ├── model1.pt
│       │   │       └── model2.pth
│       │   └── utils/
│       │       └── image_utils.py          # Image processing utilities
│       │
│       └── frontend/
│           ├── package.json                # Node.js dependencies
│           ├── vite.config.js              # Vite configuration
│           ├── tailwind.config.js          # Tailwind configuration
│           ├── postcss.config.js           # PostCSS configuration
│           ├── index.html                  # HTML entry point
│           └── src/
│               ├── main.jsx                # React entry point
│               ├── App.jsx                 # Root component
│               ├── styles.css              # Global styles
│               └── components/
│                   └── ImageGenerator.jsx  # Main UI component
│
└── README.md                               # This file
```

---

## Installation

### Prerequisites

- **Python**: 3.10 or higher
- **Node.js**: 16.x or higher
- **npm**: 8.x or higher
- **CUDA** (optional): For GPU acceleration

### Backend Setup

1. Navigate to the project root:
```bash
cd dev/
```

2. Verify Python dependencies installation:
```bash
pip install -r requirements.txt
```

3. Verify installation:
```bash
python -c "import torch; print(f'PyTorch {torch.__version__} installed')"
python -c "import flask; print(f'Flask {flask.__version__} installed')"
```

### Frontend Setup

1. Navigate to frontend directory:
```bash
cd src/DDPM_UI_FullStack/frontend/
```

2. Install Node.js dependencies:
```bash
npm install
```

3. Verify installation:
```bash
npm list react vite tailwindcss
```

---

## Running the Application

### Backend Server

Execute from the project root (`dev/`):

```bash
python -m src.DDPM_UI_FullStack.backend.app
```

Expected output:
```
2025-12-04 01:32:44,110 - __main__ - INFO - DDPM Image Generator - Initializing...
2025-12-04 01:32:44,110 - __main__ - INFO - ============================================================

GPU detected: NVIDIA GeForce RTX 4060 Laptop GPU
2025-12-04 01:32:44,136 - src.DDPM_UI_FullStack.backend.model.model_manager - INFO - Model directory initialized: /dev/src/DDPM_UI_FullStack/backend/model/SerializedModels
2025-12-04 01:32:44,136 - src.DDPM_UI_FullStack.backend.model.model_manager - INFO - Cache enabled: True
2025-12-04 01:32:44,136 - src.DDPM_UI_FullStack.backend.model.model_manager - INFO - Device: cuda
2025-12-04 01:32:44,136 - __main__ - INFO - Found 4 models:
2025-12-04 01:32:44,136 - __main__ - INFO -    - CALEBA_model_76_76_500steps_202599samples_varSelfattn_False_weights (24.57MB)
2025-12-04 01:32:44,136 - __main__ - INFO -    - cifar10_model_84_84_1000steps_60000samples_varSelfattn_False_weights (24.56MB)
2025-12-04 01:32:44,136 - __main__ - INFO -    - picobanana_model_64_64_1000steps_21896samples_varSelfattn_False_weights (24.56MB)
2025-12-04 01:32:44,136 - __main__ - INFO -    - picobanana_model_80_80_500steps_21896samples_varSelfattn_False_weights (24.56MB)
2025-12-04 01:32:44,136 - __main__ - INFO - ============================================================
2025-12-04 01:32:44,136 - __main__ - INFO - Server ready to receive requests
2025-12-04 01:32:44,136 - __main__ - INFO - ============================================================


Flask server running at: http://localhost:5000
Available endpoints:
   - GET  http://localhost:5000/ (health check)
   - GET  http://localhost:5000/models (list models)
   - POST http://localhost:5000/generate (generate image)

Debug mode: True

2025-12-04 01:32:44,137 - werkzeug - WARNING -  * Debugger is active!
2025-12-04 01:32:44,138 - werkzeug - INFO -  * Debugger PIN: 491-283-490


```

**Backend URL**: `http://localhost:5000`

### Frontend Development Server

From `src/DDPM_UI_FullStack/frontend/`:

```bash
npm run dev
```

Expected output:
```
VITE v5.0.8  ready in 234 ms

➜  Local:   http://localhost:5173/
➜  Network: use --host to expose
```

**Frontend URL**: `http://localhost:5173`

### Production Build (Frontend)

```bash
cd src/DDPM_UI_FullStack/frontend/
npm run build
npm run preview
```

---

## Model Integration

### Model Storage

Place trained PyTorch models in:
```
src/DDPM_UI_FullStack/backend/model/SerializedModels/
```

### Supported Formats

**File Extensions**: `.pt`, `.pth`

**Checkpoint Formats**:

1. **State Dictionary**:
```python
torch.save(model.state_dict(), 'model.pt')
```

2. **Full Model**:
```python
torch.save(model, 'model.pt')
```

3. **Training Checkpoint**:
```python
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'loss': loss,
}, 'checkpoint.pt')
```

All formats are automatically detected and loaded by the Model Manager.

### Model Naming

Models are identified by their filename (without extension):
- `celeba_ddpm_v1.pt` → Model name: `celeba_ddpm_v1`
- `face_gen_epoch50.pth` → Model name: `face_gen_epoch50`

---

## API Reference

### GET /

Health check endpoint.

**Response**:
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

**Status Codes**: 200 (Success)

---

### GET /models

List all available models.

**Request**:
```bash
curl http://localhost:5000/models
```

**Response**:
```json
{
  "status": "success",
  "models": [
    {
      "name": "celeba_model",
      "path": "/full/path/to/celeba_model.pt",
      "extension": ".pt",
      "size_mb": 145.23
    },
    {
      "name": "face_gen",
      "path": "/full/path/to/face_gen.pth",
      "extension": ".pth",
      "size_mb": 152.67
    }
  ],
  "count": 2
}
```

---

### POST /generate

Generate an image using a specified model.

**Request**:
```bash
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "celeba_model",
    "image_size": [64, 64],
    "channels": 3,
    "num_steps": 50
  }'
```

**Request Body Parameters**:

| Parameter    | Type         | Required | Default  | Description                          |
|-------------|--------------|----------|----------|--------------------------------------|
| `model`     | string       | Yes      | -        | Model name (without extension)       |
| `image_size`| array[int,int]| No      | [64, 64] | Output image dimensions [H, W]       |
| `channels`  | integer      | No       | 3        | Number of channels (1=Gray, 3=RGB)   |
| `num_steps` | integer      | No       | 50       | Denoising iterations                 |

**Response (Success)**:
```json
{
  "status": "success",
  "image": "iVBORw0KGgoAAAANSUhEUgAA...",
  "mode": "model",
  "model_used": "celeba_model",
  "image_size": [64, 64],
  "channels": 3
}
```

**Response (Error - Model Not Found)**:
```json
{
  "status": "error",
  "message": "Model \"invalid_model\" not found",
  "available_models": ["celeba_model", "face_gen"]
}
```

---

### GET /config

Retrieve server configuration.

**Response**:
```json
{
  "device": "cuda",
  "cache_enabled": true,
  "cache_info": {
    "cached_models": 1,
    "model_names": ["celeba_model"]
  },
  "models_directory": "/path/to/SerializedModels",
  "models_available": 2,
  "default_image_size": [64, 64],
  "default_channels": 3
}
```

---

## Image Generation Pipeline

### High-Level Process

```
1. Frontend Request
   └─> User selects model and clicks "Generate"

2. HTTP POST to /generate
   └─> JSON payload with model name and parameters

3. Backend Validation
   ├─> Validate model name (security check)
   ├─> Check model availability
   └─> Extract generation parameters

4. Model Loading
   ├─> Check model cache
   ├─> Load from disk if not cached
   └─> Move to GPU/CPU as appropriate

5. Inference Pipeline
   ├─> Initialize Gaussian noise tensor
   ├─> Iterative denoising loop (num_steps iterations)
   │   ├─> Model predicts noise at timestep t
   │   ├─> Remove predicted noise
   │   └─> Add scaled noise (except final step)
   └─> Clamp output to [0, 1]

6. Image Conversion
   ├─> Tensor → NumPy array
   ├─> Normalize to [0, 255]
   ├─> Convert to PIL Image
   └─> Encode as base64 PNG

7. Response Delivery
   └─> JSON with base64 image string

8. Frontend Rendering
   └─> Decode and display in <img> tag
```

---

## Configuration

### Backend Configuration

Edit `src/DDPM_UI_FullStack/backend/app.py`:

```python
# Model caching
ENABLE_MODEL_CACHE = True

# Default generation parameters
IMAGE_SIZE = (64, 64)
CHANNELS = 3

# Model directory
MODELS_DIR = os.path.join(
    os.path.dirname(__file__), 
    'model', 
    'SerializedModels'
)
```

### Frontend Configuration

Edit `src/DDPM_UI_FullStack/frontend/src/components/ImageGenerator.jsx`:

```javascript
// Backend API URL
const BACKEND_URL = 'http://localhost:5000'

// Default generation parameters
const defaultParams = {
  image_size: [64, 64],
  channels: 3,
  num_steps: 50
}
```

### CORS Configuration

Modify allowed origins in `app.py`:

```python
CORS(app, resources={
    r"/*": {
        "origins": [
            "http://localhost:5173", 
            "http://127.0.0.1:5173",
            "http://your-domain.com"  # Add production domain
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})
```

---

## Licenses and References

- **pico-banana-400k Dataset**

```
@misc{qian2025picobanana400klargescaledatasettextguided,
      title={Pico-Banana-400K: A Large-Scale Dataset for Text-Guided Image Editing}, 
      author={Yusu Qian and Eli Bocek-Rivele and Liangchen Song and Jialing Tong and Yinfei Yang and Jiasen Lu and Wenze Hu and Zhe Gan},
      year={2025},
      eprint={2510.19808},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.19808}, 
}
```

- **CELEBA Dataset**
```
@inproceedings{liu2015faceattributes,
  title = {Deep Learning Face Attributes in the Wild},
  author = {Liu, Ziwei and Luo, Ping and Wang, Xiaogang and Tang, Xiaoou},
  booktitle = {Proceedings of International Conference on Computer Vision (ICCV)},
  month = {December},
  year = {2015} 
}
```

- **CIFAR10 Dataset**

> [Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf), Alex Krizhevsky, 2009.

- **Articles and Reference Implementations**

  - [DDPM from scratch in Pytorch](https://www.kaggle.com/code/vikramsandu/ddpm-from-scratch-in-pytorch#Diffusion-Model---The-Structure)

  - [U-Net Architecture Explained](https://www-geeksforgeeks-org.translate.goog/machine-learning/u-net-architecture-explained/?_x_tr_sl=en&_x_tr_tl=es&_x_tr_hl=es&_x_tr_pto=tc)

  - [DDPM PyTorch Implementation from Scratch](https://medium.com/@sayedebad.777/ddpm-pytorch-implementation-from-scratch-36b647f5dd82)


# Contact
This project was developed by:
- Alyson Melissa Sánchez Serratos
- Miguel Ángel Pérez Ávila
 