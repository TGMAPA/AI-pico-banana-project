from src.config.libraries import *

# Forward diffusion formula as in paper DDPM 
# x_t = sqrt(alpha_product_acum) * x_0  +  sqrt(1 - alpha_product_acum) * noise

class DiffusionFwd:
    def __init__(self, N_T_steps = 1000, beta_0 = 1e-4, beta_N = 0.02):
        # Define betas for each t steps increasing by N_T_Steps 
        self.betas = torch.linspace(beta_0, beta_N, N_T_steps)

        # Define alphas as 	α = 1 - β
        self.alphas = 1 - self.betas

        # Cumulative product of α for each alpha_t
        self.alpha_product_acum = torch.cumprod(self.alphas, dim=0)

        # Precompute the coefficients used in the closed-form forward diffusion equation

        # The remaining signal of the original image at timestep t
        self.sqrt_alpha_product_acum = torch.sqrt(self.alpha_product_acum) 
        
        # The noise scale at timestep t
        self.sqrt_one_minus_alpha_product_acum = torch.sqrt(1 - self.alpha_product_acum)

    # Add noise to a barch of original images at a specific time step t
    def createNoise(self, images_batch, noise, time_step):
        # Select the precomputed coefficients for the specific timesteps of each image in the batch
        sqrt_alpha_product_acum_t = self.sqrt_alpha_product_acum.to(images_batch.device)[time_step]
        
        sqrt_one_minus_alpha_acum_t = self.sqrt_one_minus_alpha_product_acum.to(images_batch.device)[time_step]
        
        # Broadcast to multiply with the original image.
        sqrt_alpha_product_acum_t = sqrt_alpha_product_acum_t.view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_acum_t = sqrt_one_minus_alpha_acum_t.view(-1, 1, 1, 1)

        # Apply the forward diffusion formula, returning processed image x_t = sqrt(alpha_product_acum) * x_0  +  sqrt(1 - alpha_product_acum) * noise
        return (sqrt_alpha_product_acum_t * images_batch) + (sqrt_one_minus_alpha_acum_t * noise)
        


def test_diffusion_forward():
    # Parámetros de la prueba
    batch_size = 4
    channels = 3
    height = 64
    width = 64
    t = 100  # timestep a probar

    # Instanciar difusión
    diffusion = DiffusionFwd(N_T_steps=1000, beta_0=1e-4, beta_N=0.02)

    # Crear un batch aleatorio de imágenes
    images = torch.randn(batch_size, channels, height, width).to("cpu")

    # Crear ruido aleatorio
    noise = torch.randn_like(images).to("cpu")

    # Aplicar difusión
    noisy_images = diffusion.createNoise(images, noise, t)

    # -----------------------
    # VERIFICACIONES
    # -----------------------
    print("Shapes:")
    print(f"  images       : {images.shape}")
    print(f"  noise        : {noise.shape}")
    print(f"  noisy_images : {noisy_images.shape}")

    # Deben coincidir
    assert noisy_images.shape == images.shape, "Error: output shape mismatch"

    # Comprobar que no son iguales → debe haber ruido agregado
    diff = (noisy_images - images).abs().mean().item()
    print(f"\nMean absolute difference (should be > 0): {diff}")

    # Revisar un valor concreto de los coeficientes
    print("\nCoeficientes en t:")
    print(" sqrt(alpha_prod):", diffusion.sqrt_alpha_product_acum[t].item())
    print(" sqrt(1 - alpha_prod):", diffusion.sqrt_one_minus_alpha_product_acum[t].item())

    print("\nTest passed.\n")


test_diffusion_forward()