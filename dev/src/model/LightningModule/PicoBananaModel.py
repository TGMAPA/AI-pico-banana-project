# Pico banana model

# Import required modules and libraries
from src.model.DiffusionFwd.DiffusionFwd import DiffusionFwd
from src.config.libraries import *
from src.config.config import DEVICE, N_T_STEPS, BETA_0, BETA_N


# Picobanana Model Condiguration with Lightning tools
class PicobananaModel(L.LightningModule):
    # Class constructor
    def __init__(self, model, learning_rate=1e-4, num_timesteps=1000):
        super().__init__()
        # General properties
        self.learning_rate = learning_rate
        self.model = model
        self.num_timesteps = num_timesteps

        # Define loss function
        self.loss_fn = nn.MSELoss()

        # Diffusion Forward Process to add noise
        self.difussionForward = DiffusionFwd(
            N_T_steps = N_T_STEPS, 
            beta_0 = BETA_0, 
            beta_N = BETA_N
        )

    # Compute Model Forward
    def forward(self, x, t):
        noise_prediction = self.model(x, t)
        return noise_prediction
    
    # Compute DDPM steps 
    def shared_step(self, batch):
        images = batch
        
        # --- Direct Difussion Forward (Noise adition)
        # Generate noise and timestaps
        noise = torch.randn_like(images).to(DEVICE)
        t = torch.randint(0, self.num_timesteps, (images.shape[0],)).to(DEVICE)
    
        # Add noise with Diffusion Forward process
        noisy_images = self.difussionForward.createNoise(images, noise, t)

        # --- Get prediction
        # Get noise prediction with model for actual batch
        noise_prediction = self(noisy_images, t)

        # --- Loss
        # Compute error between original noise distribution and model's noise prediction for actual batch
        loss_model = self.loss_fn(noise_prediction, noise)

        return loss_model
    
    # Execute training step and store loss 
    def training_step(self, batch, batch_idx):
        loss_model = self.shared_step(batch)
        
        # Log computed loss
        self.log("train_loss", loss_model, on_epoch=True, logger=True)
        
        # --- Accuracy Computation --- Pending

        return loss_model
    
    # Execute training step and store loss 
    def validation_step(self, batch, batch_idx):
        loss_model = self.shared_step(batch)
        
        # Log computed loss
        self.log("val_loss", loss_model, on_epoch=True, logger=True)
        
        # --- Accuracy Computation --- Pending

        return loss_model

    # Configure Model Optimizer
    def configure_optimizers(self):
        # Configure optimizer as ADAM with specified learning rate
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
