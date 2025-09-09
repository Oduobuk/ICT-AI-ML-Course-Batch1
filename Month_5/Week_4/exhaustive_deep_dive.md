# Month 5, Week 4: Exhaustive Deep Dive - Generative Models

## Generative Adversarial Networks (GANs) - In-depth

### Architecture and Training:
GANs consist of two neural networks, the Generator (G) and the Discriminator (D), that compete against each other in a zero-sum game.

- **Generator (G)**: A neural network that takes a random noise vector $z$ (typically sampled from a simple distribution like a uniform or Gaussian distribution) as input and transforms it into a synthetic data sample $G(z)$. The goal of G is to learn the data distribution of the real samples.

- **Discriminator (D)**: A neural network that takes an input sample (either a real data sample $x$ or a synthetic sample $G(z)$) and outputs a probability indicating whether the input is real or fake. The goal of D is to correctly distinguish between real and fake samples.

### Objective Function:
The training of a GAN is formulated as a minimax game with the following value function $V(D, G)$:

$\\min_G \\max_D V(D, G) = \\mathbb{E}_{x \\sim p_{data}(x)}[\\log D(x)] + \\mathbb{E}_{z \\sim p_z(z)}[\\log(1 - D(G(z)))]$

Where:
- $p_{data}(x)$ is the real data distribution.
- $p_z(z)$ is the noise distribution.
- $\\mathbb{E}$ denotes the expectation.

During training:
- D is trained to maximize $V(D, G)$, meaning it wants to correctly classify real samples as real (output close to 1) and fake samples as fake (output close to 0).
- G is trained to minimize $V(D, G)$, meaning it wants to generate samples that D classifies as real (output close to 1).

### Training Challenges:
- **Mode Collapse**: The generator might produce a limited variety of samples, collapsing to a few modes of the data distribution.
- **Vanishing Gradients**: If the discriminator becomes too good too early, the generator's gradients can vanish, hindering its learning.
- **Training Instability**: Oscillations and non-convergence are common due to the adversarial nature of the training.

### Types of GANs:
- **DCGAN (Deep Convolutional GAN)**: Uses convolutional layers without pooling or fully connected layers, and uses batch normalization, leading to more stable training and better image generation.
- **Conditional GAN (cGAN)**: Allows for controlled data generation by conditioning both the generator and discriminator on some auxiliary information (e.g., class labels, text descriptions).
- **Wasserstein GAN (WGAN)**: Uses the Wasserstein distance (Earth Mover's distance) as a loss function, which provides a more stable gradient and helps mitigate mode collapse.

## Variational Autoencoders (VAEs) - Detailed Exploration

### Architecture:
VAEs are generative models that learn a probabilistic mapping from data to a latent space and back. They consist of an encoder and a decoder.

- **Encoder**: Maps the input data $x$ to a probability distribution over the latent space. Instead of outputting a single latent vector, it outputs the parameters (mean $\\mu$ and variance $\\sigma^2$) of a Gaussian distribution $q_{\\phi}(z|x)$.

- **Reparameterization Trick**: To allow backpropagation through the sampling process, a sample $z$ from $q_{\\phi}(z|x)$ is obtained as $z = \\mu + \\sigma \\cdot \\epsilon$, where $\\epsilon \\sim \\mathcal{N}(0, 1)$. This separates the stochasticity from the deterministic part of the network.

- **Decoder**: Takes a sample $z$ from the latent space and reconstructs the input data $p_{\\theta}(x|z)$.

### Loss Function (Evidence Lower Bound - ELBO):
The VAE objective function is to maximize the Evidence Lower Bound (ELBO), which is a lower bound on the marginal likelihood of the data. It consists of two terms:

1.  **Reconstruction Loss**: Measures how well the decoder reconstructs the input data. Typically, this is the negative log-likelihood of the data given the reconstructed output (e.g., Mean Squared Error for continuous data, Binary Cross-Entropy for binary data).
    $\\mathbb{E}_{z \\sim q_{\\phi}(z|x)}[\\log p_{\\theta}(x|z)]$

2.  **KL Divergence Loss**: Measures the difference between the learned latent distribution $q_{\\phi}(z|x)$ and a prior distribution $p(z)$ (usually a standard normal distribution $\\mathcal{N}(0, 1)$). This term acts as a regularizer, encouraging the latent space to be well-behaved.
    $D_{KL}(q_{\\phi}(z|x) || p(z))$

The total loss to minimize is: $-\\mathbb{E}_{z \\sim q_{\\phi}(z|x)}[\\log p_{\\theta}(x|z)] + D_{KL}(q_{\\phi}(z|x) || p(z))$

### Advantages:
- **Structured Latent Space**: The KL divergence term encourages the latent space to be continuous and well-structured, allowing for meaningful interpolation and sampling.
- **Probabilistic Framework**: Provides a probabilistic interpretation of data generation.
- **Applications**: Image generation, anomaly detection, data imputation, semi-supervised learning.

## Other Generative Models - Deeper Dive

### Autoregressive Models:
- **Concept**: Model the joint probability distribution of data as a product of conditional probabilities. For example, in an image, each pixel is generated conditioned on previously generated pixels.
- **Examples**: PixelRNN, PixelCNN, WaveNet (for audio).
- **Pros**: Can model complex distributions and provide exact likelihoods.
- **Cons**: Sequential generation makes them slow for high-dimensional data; lack of parallelization during sampling.

### Flow-based Models:
- **Concept**: Construct complex probability distributions by transforming a simple base distribution (e.g., Gaussian) through a sequence of invertible and differentiable transformations. The change of variables formula is used to compute the likelihood.
- **Examples**: NICE (Non-linear Independent Components Estimation), Real NVP (Real-valued Non-Volume Preserving transformations), Glow.
- **Pros**: Exact likelihood computation, efficient sampling and inference, invertible transformations allow for both generation and density estimation.
- **Cons**: Can be computationally expensive for very deep flows; architectural constraints to ensure invertibility.

## Further Reading and References:
- **GANs**: *"Generative Adversarial Nets"* by Ian Goodfellow et al. (2014).
- **VAEs**: *"Auto-Encoding Variational Bayes"* by Diederik P. Kingma and Max Welling (2013).
- **DCGAN**: *"Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks"* by Alec Radford et al. (2015).
- **WGAN**: *"Wasserstein GAN"* by Martin Arjovsky et al. (2017).
- **Flow-based Models**: *"Density estimation using Real NVP"* by Laurent Dinh et al. (2016).

This exhaustive deep dive provides a comprehensive understanding of generative models, their mathematical underpinnings, and their applications in various domains.
