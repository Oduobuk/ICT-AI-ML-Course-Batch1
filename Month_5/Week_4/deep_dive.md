# Month 5, Week 4: Deep Dive - Generative Models

## Generative Adversarial Networks (GANs)
- **Concept**: A framework for estimating generative models via an adversarial process, involving two neural networks: a Generator (G) and a Discriminator (D).
- **Generator (G)**: Takes random noise as input and tries to generate data that resembles the real data distribution.
- **Discriminator (D)**: Takes both real data and generated data as input and tries to distinguish between them.
- **Training Process**: G and D are trained simultaneously in a minimax game. G tries to fool D, and D tries to correctly classify real vs. fake.
- **Challenges**: Mode collapse, training instability, difficulty in evaluating generated samples.

## Variational Autoencoders (VAEs)
- **Concept**: A generative model that learns a probabilistic mapping from data to a latent space and back.
- **Encoder**: Maps input data to a distribution (mean and variance) in the latent space.
- **Reparameterization Trick**: Allows backpropagation through the sampling process from the latent distribution.
- **Decoder**: Reconstructs the input data from a sample drawn from the latent distribution.
- **Loss Function**: Combination of reconstruction loss (how well the data is reconstructed) and KL divergence (how close the latent distribution is to a prior, usually a standard normal distribution).
- **Advantages**: Provides a structured latent space, allows for controlled generation and interpolation.

## Other Generative Models
- **Autoregressive Models (e.g., PixelRNN, WaveNet)**: Generate data one element at a time, conditioned on previously generated elements. Good for sequential data but slow for high-dimensional data.
- **Flow-based Models (e.g., NICE, Real NVP)**: Learn an invertible transformation that maps a simple distribution (e.g., Gaussian) to the data distribution. Allow for exact likelihood computation and efficient sampling.

## Key Concepts:
- **Latent Space**: A lower-dimensional representation of the data, capturing its underlying structure.
- **Generative vs. Discriminative Models**: Generative models learn the distribution of data, while discriminative models learn to classify data.
- **Likelihood**: The probability of observing the data given the model parameters.
- **Sampling**: Generating new data points from the learned distribution.