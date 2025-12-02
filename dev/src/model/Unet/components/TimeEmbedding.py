# Import requiered modules
from src.config.libraries import *


# Generate time embedding from escalar t value
class TimeEmbedding(nn.Module):
    # Class constructor
    def __init__(self, time_embedding_dimension, fc_out_dimension):
        super().__init__()
        
        # Verify specified time embedding's dimension
        assert time_embedding_dimension % 2 == 0, "Time Embedding desired dimension is not divisible by 2."

        # Desired dimensions for time embedding
        self.time_embedding_dimension = time_embedding_dimension

        # Desired fully connected output's dimensions
        self.fc_out_dimension = fc_out_dimension

        # Fully conected layer to transform time embedding
        self.fc = nn.Sequential(
            nn.Linear(self.time_embedding_dimension, self.time_embedding_dimension),
            # SiLU is used because it provides smooth, stable gradients and works very well
            # with sinusoidal time embeddings.
            nn.SiLU(),
            nn.Linear(self.time_embedding_dimension, self.fc_out_dimension)
        )

    # Transform escalar t into expanded time embedding 
    def generate_time_embedding(self, time_steps):
        # Get half of the time_embedding_dimension
        half = self.time_embedding_dimension//2

        # Compute frequencies tensor as freq_i = 10000 ** (i / (D/2))
        I = torch.arange(0, half, device= time_steps.device)
        I_div_halfTimeDim = I / half
        freq = 10000 ** I_div_halfTimeDim  # (D/2, )

        # Convert timesteps from shape (B,) to (B, 1), then repeat each timestep 'half' times
        # so we obtain a matrix of shape (B, half)
        expanded_time_steps = time_steps.float().unsqueeze(1).repeat(1, half)

        # Divide expanded timestep by the frequency vector
        timesteps_div_freq = expanded_time_steps / freq

        # Apply Sine function for the first half of the embedding
        sine_time_embedding = torch.sin(timesteps_div_freq)

        # Apply Cosine function for the second half of the embedding
        cosine_time_embedding = torch.cos(timesteps_div_freq)

        # Concatenate Sine + Cosine and build final time embedding tensor
        time_embedding = torch.cat([sine_time_embedding, cosine_time_embedding], dim= -1)
        
        return time_embedding
    
    # Execute forward step 
    def forward(self, time_steps):
        timeEmbedding = self.generate_time_embedding(time_steps)
        timeEmbedding = self.fc(timeEmbedding)

        return timeEmbedding
