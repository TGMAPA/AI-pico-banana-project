"""
Flask Backend Server for Dynamic DDPM Model Generation
========================================================
Flask server exposing endpoints for dynamic model selection and image generation.

Endpoints:
    - GET  /: Health check
    - GET  /models: List available models
    - POST /generate: Generate image with selected model
    
Features:
    - Dynamic model loading from SerializedModels directory
    - Model validation and caching
    - Flexible model selection via request body
"""

# Import local modules
from src.config.libraries import *
from src.DDPM_UI_FullStack.backend.model.model_manager import ModelManager, ModelValidator
from src.DDPM_UI_FullStack.backend.model.ddpm_inference import generate_image, create_sample_image
from src.DDPM_UI_FullStack.backend.utils.image_utils import pil_to_base64


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Configure CORS for React (port 5173)
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:5173", "http://127.0.0.1:5173"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Global variables
MODEL_MANAGER = None
DEVICE = None
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'model', 'SerializedModels')

# Configuration
ENABLE_MODEL_CACHE = True
IMAGE_SIZE = (64, 64)
CHANNELS = 3


def initialize_server():
    """
    Initialize the server with ModelManager and device detection.
    """
    global MODEL_MANAGER, DEVICE
    
    logger.info("\n" + "="*60)
    logger.info("DDPM Image Generator - Initializing...")
    logger.info("="*60 + "\n")
    
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        DEVICE = torch.device("cpu")
        print("GPU not available, using CPU")
    
    # Initialize ModelManager
    try:
        MODEL_MANAGER = ModelManager(
            models_dir=MODELS_DIR,
            enable_cache=ENABLE_MODEL_CACHE,
            device=DEVICE
        )
        
        # List available models
        available_models = MODEL_MANAGER.list_available_models()
        
        if available_models:
            logger.info(f"Found {len(available_models)} models:")
            for model in available_models:
                logger.info(f"   - {model['name']} ({model['size_mb']}MB)")
        else:
            logger.warning("No models found in SerializedModels directory")
        
        logger.info("="*60)
        logger.info("Server ready to receive requests")
        logger.info("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"Failed to initialize ModelManager: {str(e)}")
        raise


@app.route('/', methods=['GET'])
def health_check():
    """
    Health check endpoint to verify server is running.
    
    Returns:
        JSON with server status and configuration
    """
    available_models = MODEL_MANAGER.list_available_models() if MODEL_MANAGER else []
    
    return jsonify({
        'status': 'online',
        'models_available': len(available_models),
        'device': str(DEVICE),
        'cache_enabled': ENABLE_MODEL_CACHE,
        'image_size': IMAGE_SIZE,
        'channels': CHANNELS
    })


@app.route('/models', methods=['GET'])
def list_models():
    """
    Get list of all available models.
    
    Returns:
        JSON with array of available models
        Format: [
            {
                'name': 'model_name',
                'path': '/full/path/to/model.pt',
                'extension': '.pt',
                'size_mb': 123.45
            },
            ...
        ]
    """
    try:
        if not MODEL_MANAGER:
            return jsonify({
                'status': 'error',
                'message': 'ModelManager not initialized'
            }), 500
        
        available_models = MODEL_MANAGER.list_available_models()
        
        return jsonify({
            'status': 'success',
            'models': available_models,
            'count': len(available_models)
        })
    
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/generate', methods=['POST'])
def generate():
    """
    Generate image with selected model.
    
    Request Body (JSON):
        {
            'model': 'model_name',           # Required: name of model to use
            'image_size': [64, 64],         # Optional: image dimensions
            'channels': 3,                   # Optional: 3=RGB, 1=Grayscale
            'num_steps': 50                 # Optional: denoising steps
        }
    
    Response (JSON):
        {
            'image': '<base64_encoded_png>',
            'status': 'success',
            'mode': 'model' or 'sample',
            'model_used': 'model_name',
            'image_size': [64, 64],
            'channels': 3
        }
    """
    try:
        if not MODEL_MANAGER:
            return jsonify({
                'status': 'error',
                'message': 'ModelManager not initialized'
            }), 500
        
        # Parse request body
        data = request.get_json() or {}
        
        # Get model name from request
        model_name_raw = data.get('model')
        model_name_raw = model_name_raw.split("_")
        
        # Extract image_size for the model
        image_size = [int(model_name_raw[2]), int(model_name_raw[3])]
        
        # Extract number of steps for image generation
        num_steps = int(model_name_raw[4].split("steps")[0])

        # Extract model_name
        model_name = data.get('model')
        
        # Get optional parameters
        channels = data.get('channels', CHANNELS)
        
        logger.info("\nImage generation request received:")
        logger.info(f"   - Model: {model_name}")
        logger.info(f"   - Size: {image_size}")
        logger.info(f"   - Channels: {channels}")
        logger.info(f"   - Steps: {num_steps}")
        
        # Validate model name
        if not model_name:
            logger.warning("Model name not provided")
            return jsonify({
                'status': 'error',
                'message': 'Model name is required'
            }), 400
        
        # Validate model name format
        is_valid, error_msg = ModelValidator.validate_model_name(model_name)
        if not is_valid:
            logger.warning(f"Invalid model name: {error_msg}")
            return jsonify({
                'status': 'error',
                'message': f'Invalid model name: {error_msg}'
            }), 400
        
        # Check if model exists
        if not MODEL_MANAGER.model_exists(model_name):
            logger.warning(f"Model not found: {model_name}")
            available_models = [m['name'] for m in MODEL_MANAGER.list_available_models()]
            return jsonify({
                'status': 'error',
                'message': f'Model "{model_name}" not found',
                'available_models': available_models
            }), 400
        
        try:
            # Load model dynamically
            logger.info(f"Loading model: {model_name}")
            model, device = MODEL_MANAGER.load_model(model_name)
            
            # Generate image --------------------------------------
            logger.info(f"Generating image with model: {model_name}")
            image = generate_image(
                model=model,
                device=device,
                image_size=image_size,
                channels=channels,
                num_steps=num_steps
            )
            mode = "model"
            
        except FileNotFoundError as e:
            logger.error(f"Model file not found: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': f'Model file not found: {str(e)}'
            }), 404
        except Exception as e:
            logger.error(f"Error loading/using model: {str(e)}")
            # Fallback to sample image on model error
            logger.info("Falling back to sample image mode")
            image = create_sample_image(
                device=DEVICE,
                image_size=image_size,
                channels=channels
            )
            mode = "sample"
        
        # Transform ndarray C,H,W → H,W,C
        image = np.transpose(image, (1, 2, 0))

        # Transform to PIL
        pil_image = Image.fromarray(image)

        # Convert image to base64
        logger.info("Converting image to base64...")
        image_base64 = pil_to_base64(pil_image, format='PNG')
        
        logger.info(f"Image generated successfully ({mode} mode)")
        logger.info(f"   - Base64 size: {len(image_base64)} characters\n")
        
        return jsonify({
            'image': image_base64,
            'status': 'success',
            'mode': mode,
            'model_used': model_name,
            'image_size': list(image_size),
            'channels': channels
        })
    
    except Exception as e:
        logger.error(f"Error in generate endpoint: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


if __name__ == '__main__':
    try:
        # Initialize server
        initialize_server()
        
        # Server configuration
        PORT = 5000
        DEBUG = True
        
        print(f"\nFlask server running at: http://localhost:{PORT}")
        print(f"Available endpoints:")
        print(f"   - GET  http://localhost:{PORT}/ (health check)")
        print(f"   - GET  http://localhost:{PORT}/models (list models)")
        print(f"   - POST http://localhost:{PORT}/generate (generate image)")
        print(f"\nDebug mode: {DEBUG}\n")
        
        # Start server
        app.run(
            host='0.0.0.0',
            port=PORT,
            debug=DEBUG
        )
    
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        sys.exit(1)
