"""
Model Manager Module
====================
Centralized management for dynamic model loading, caching, and validation.

Features:
    - Automatic model discovery from SerializedModels directory
    - Dynamic model loading with validation
    - Optional caching to avoid redundant loads
    - Error handling for missing or corrupted models
"""

# Import local modules
from src.config.libraries import *
from src.model.PicoBanana import PicoBanana

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages model loading, caching, and validation.
    
    Attributes:
        models_directory: Path to SerializedModels directory
        model_cache: Dictionary storing loaded models
        enable_cache: Whether to cache models in memory
    """
    
    SUPPORTED_EXTENSIONS = ['.pt', '.pth']
    
    def __init__(self, models_dir: str, enable_cache: bool = True, device: Optional[torch.device] = None):
        """
        Initialize ModelManager.
        
        Args:
            models_dir: Path to SerializedModels directory
            enable_cache: Whether to cache loaded models
            device: torch.device (cuda or cpu). Auto-detected if None
        """
        self.models_directory = Path(models_dir)
        self.model_cache: Dict[str, torch.nn.Module] = {}
        self.enable_cache = enable_cache
        self.device = device or self._detect_device()
        
        # Ensure directory exists
        self.models_directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Model directory initialized: {self.models_directory}")
        logger.info(f"Cache enabled: {self.enable_cache}")
        logger.info(f"Device: {self.device}")
    
    @staticmethod
    def _detect_device() -> torch.device:
        """Auto-detect available device (GPU or CPU)."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            logger.info("GPU not available, using CPU")
        return device
    
    def list_available_models(self) -> List[Dict[str, str]]:
        """
        List all available models in SerializedModels directory.
        
        Returns:
            List of dicts with model metadata (name, path, extension, size_mb)
        """
        models = []
        
        if not self.models_directory.exists():
            logger.warning(f"Models directory not found: {self.models_directory}")
            return models
        
        for ext in self.SUPPORTED_EXTENSIONS:
            for model_file in self.models_directory.glob(f"*{ext}"):
                model_info = {
                    'name': model_file.stem,  # Model name without extension
                    'path': str(model_file),
                    'extension': ext,
                    'size_mb': round(model_file.stat().st_size / (1024 * 1024), 2)
                }
                models.append(model_info)
        
        logger.debug(f"Found {len(models)} models")
        for model in models:
            logger.debug(f"   - {model['name']} ({model['size_mb']}MB)")
        
        return sorted(models, key=lambda x: x['name'])
    
    def model_exists(self, model_name: str) -> bool:
        """
        Check if a model exists in the directory.
        
        Args:
            model_name: Name of the model (without extension)
        
        Returns:
            True if model exists, False otherwise
        """
        models = self.list_available_models()
        return any(m['name'] == model_name for m in models)
    
    def get_model_path(self, model_name: str) -> Optional[str]:
        """
        Get the full path to a model file.
        
        Args:
            model_name: Name of the model (without extension)
        
        Returns:
            Full path to model file or None if not found
        """
        models = self.list_available_models()
        for model in models:
            if model['name'] == model_name:
                return model['path']
        return None
    
    def load_model(self, model_name: str) -> Tuple[Optional[torch.nn.Module], torch.device]:
        """
        Load a model by name. Uses cache if enabled.
        
        Args:
            model_name: Name of the model (without extension)
        
        Returns:
            Tuple of (model, device) or (None, device) if loading fails
        
        Raises:
            FileNotFoundError: If model doesn't exist
            RuntimeError: If model loading fails
        """
        # Check cache first
        if self.enable_cache and model_name in self.model_cache:
            logger.info(f"Loading '{model_name}' from cache")
            return self.model_cache[model_name], self.device
        
        # Get model path
        model_path = self.get_model_path(model_name)
        if not model_path:
            raise FileNotFoundError(f"Model '{model_name}' not found in {self.models_directory}")
        
        try:
            logger.debug(f"Loading model from: {model_path}")
            
            # Create model
            model = PicoBanana(
                batch_size = 64,
                num_workers = 16,
                load_dm = False
            )

            # Save model
            model.load_model( model_path )
            print("Model was correctly loaded from " +model_path+ " ...")
            
            # Store in cache if enabled
            if self.enable_cache:
                self.model_cache[model_name] = model
                logger.debug(f"Model cached: {model_name}")
            
            logger.info(f"Model '{model_name}' loaded successfully")
            return model, self.device
        
        except Exception as e:
            logger.error(f"Failed to load model '{model_name}': {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def clear_cache(self) -> None:
        """Clear the model cache to free memory."""
        self.model_cache.clear()
        logger.info("Model cache cleared")
    
    def get_cache_info(self) -> Dict[str, int]:
        """Get information about cached models."""
        return {
            'cached_models': len(self.model_cache),
            'model_names': list(self.model_cache.keys())
        }


class ModelValidator:
    """Validates models before loading."""
    
    @staticmethod
    def validate_model_name(model_name: str) -> Tuple[bool, str]:
        """
        Validate model name format.
        
        Args:
            model_name: Name to validate
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not model_name:
            return False, "Model name cannot be empty"
        
        if len(model_name) > 255:
            return False, "Model name too long"
        
        # Check for invalid characters
        invalid_chars = ['/', '\\', '..', '\x00']
        for char in invalid_chars:
            if char in model_name:
                return False, f"Model name contains invalid character: {char}"
        
        return True, ""
    
    @staticmethod
    def validate_checkpoint_format(checkpoint):
        """
        Validate if checkpoint has expected format.
        
        Args:
            checkpoint: Loaded checkpoint object
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if isinstance(checkpoint, dict):
            # Check for expected keys
            expected_keys = ['model_state_dict', 'state_dict', 'model']
            if not any(key in checkpoint for key in expected_keys):
                return False, "Checkpoint missing expected state dict keys"
        
        return True, ""
