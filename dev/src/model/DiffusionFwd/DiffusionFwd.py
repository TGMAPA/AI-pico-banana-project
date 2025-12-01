# Import libraries
from src.config.libraries import *

# Forward diffusion formula as in paper DDPM 
# x_t = sqrt(alpha_product_acum) * x_0  +  sqrt(1 - alpha_product_acum) * noise

class DiffusionFwd:
    def __init__(self, N_T_steps = 1000, beta_0 = 1e-4, beta_N = 0.02):
        # Define betas for each t steps increasing by N_T_Steps 
        self.betas = torch.linspace(beta_0, beta_N, N_T_steps) # shape: (T_steps,)

        # Define alphas as 	α = 1 - β
        self.alphas = 1 - self.betas # shape: (T_steps,)

        # Cumulative product of α for each alpha_t
        self.alpha_product_acum = torch.cumprod(self.alphas, dim=0)  # shape: (T_steps,)

        # Precompute the coefficients used in the closed-form forward diffusion equation

        # The remaining signal of the original image at timestep t
        self.sqrt_alpha_product_acum = torch.sqrt(self.alpha_product_acum) # shape: (T_steps,)
        
        # The noise scale at timestep t
        self.sqrt_one_minus_alpha_product_acum = torch.sqrt(1 - self.alpha_product_acum) # shape: (T_steps,)

    # Add noise to a barch of original images at a specific time step t
    def createNoise(self, images_batch, noise, time_step):
        
        """
        images_batch:  -> shape (Batch, Channels, Height, Width)
        time_step:     -> shape (Batch,) with integers in [0, T_steps)
        noise:         -> shape (Batch, Channels, Height, Width) with Gaussian noise N(0, 1)
        """
               
        # Select the precomputed coefficients for the specific timesteps of each image in the batch
        sqrt_alpha_product_acum_t = self.sqrt_alpha_product_acum.to(images_batch.device)[time_step] # shape: (Batch,)
        
        sqrt_one_minus_alpha_acum_t = self.sqrt_one_minus_alpha_product_acum.to(images_batch.device)[time_step] # shape: (Batch,)
        
        # Broadcast to multiply with the original image
        # (Batch,) -> (Batch, 1, 1, 1)
        sqrt_alpha_product_acum_t = sqrt_alpha_product_acum_t.view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_acum_t = sqrt_one_minus_alpha_acum_t.view(-1, 1, 1, 1)

        # Apply the forward diffusion formula, returning processed image x_t = sqrt(alpha_product_acum) * x_0  +  sqrt(1 - alpha_product_acum) * noise
        return (sqrt_alpha_product_acum_t * images_batch) + (sqrt_one_minus_alpha_acum_t * noise) # shape: (Batch, Channels, Height, Width)

    # Test the forward diffusion process by adding noise to a randomly generated image batch using a chosen timestep t within the total number of diffusion steps T.
    def test_diffusion_forward(self, batch_size = 4, img_channels = 3, img_height = 64, img_width = 64, time_step = 100, device = "cpu"):
        # Instance Difussion Fwd Object with specified params
        diffusion = self(N_T_steps=1000, beta_0=1e-4, beta_N=0.02)

        # Crear random image batch 
        images = torch.randn(batch_size, img_channels, img_height, img_width).to(device)

        # Create random noise
        noise = torch.randn_like(images).to(device)

        # Apply difussion
        noisy_images = diffusion.createNoise(images, noise, time_step)

        # Show verifications
        print("Shapes:")
        print(f"  images       : {images.shape}")
        print(f"  noise        : {noise.shape}")
        print(f"  noisy_images : {noisy_images.shape}")
        print("\Coefficients at timestep = t:")
        print("  sqrt(alpha_prod):", diffusion.sqrt_alpha_product_acum[time_step].item())
        print("  sqrt(1 - alpha_prod):", diffusion.sqrt_one_minus_alpha_product_acum[time_step].item())

        return True