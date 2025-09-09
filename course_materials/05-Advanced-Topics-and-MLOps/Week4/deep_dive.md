
# Month 5, Week 4: Generative Models (Introduction) - Deep Dive

## Autoencoders (AE) and Variational Autoencoders (VAE)

*   **Autoencoder:** A type of neural network that is used for unsupervised learning. It consists of two parts: an encoder and a decoder. The encoder maps the input data to a lower-dimensional representation, and the decoder maps the lower-dimensional representation back to the original data.
*   **Variational Autoencoder (VAE):** A type of autoencoder that is used for generative modeling. It works by learning a probability distribution over the latent space, which allows it to generate new data points that are similar to the training data.

## Generative Adversarial Networks (GANs)

*   **GAN:** A type of generative model that consists of two neural networks: a generator and a discriminator. The generator tries to generate realistic data, and the discriminator tries to distinguish between real data and fake data.
*   **Generator/Discriminator Dynamic:** The generator and the discriminator are trained in an adversarial manner. The generator tries to fool the discriminator, and the discriminator tries to not be fooled. This process results in the generator learning to generate realistic data.
