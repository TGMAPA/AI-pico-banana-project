# Import libraries and modules
from src.config.libraries import *


# Reversed diffusion formula as in DDPM paper
class DiffusionReversed:
    def __init__(self, N_T_steps = 1000, beta_0 = 1e-4, beta_N = 0.02):
        # Define betas for each t steps increasing by N_T_Steps 
        self.betas = torch.linspace(beta_0, beta_N, N_T_steps) # shape: (T_steps,)

        # Define alphas as 	α = 1 - β
        self.alphas = 1 - self.betas # shape: (T_steps,)

        # Cumulative product of α for each alpha_t
        self.alpha_product_acum = torch.cumprod(self.alphas, dim=0)  # shape: (T_steps,)

    def reverse_timestep(self, images_batch_t, noise_prediction, time_step):
        """
        images_batch_t: Image tensor at timestep t of shape -> Batch x Channels x Height x Width
        noise_prediction: Noise Predicted by model of shape -> Batch x Channels x Height x Width
        time_step: Current time step

        """
        
         # Original Image Prediction at timestep t
        x0 = images_batch_t - (torch.sqrt(1 - self.alpha_product_acum.to(images_batch_t.device)[time_step]) * noise_prediction)
        x0 = x0/torch.sqrt(self.alpha_product_acum.to(images_batch_t.device)[time_step])
        x0 = torch.clamp(x0, -1., 1.) 
        
        # mean of x_(t-1)
        mean = (images_batch_t - ((1 - self.alphas.to(images_batch_t.device)[time_step]) * noise_prediction)/(torch.sqrt(1 - self.alpha_product_acum.to(images_batch_t.device)[time_step])))
        mean = mean/(torch.sqrt(self.alphas.to(images_batch_t.device)[time_step]))
        
        # Return mean
        if time_step == 0:
            return mean, x0
        
        else:
            variance =  (1 - self.alpha_product_acum.to(images_batch_t.device)[time_step-1])/(1 - self.alpha_product_acum.to(images_batch_t.device)[time_step])
            variance = variance * self.betas.to(images_batch_t.device)[time_step]
            sigma = variance**0.5
            z = torch.randn(images_batch_t.shape).to(images_batch_t.device)
            
            return mean + sigma * z, x0