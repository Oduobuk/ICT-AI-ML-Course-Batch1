# Month 3, Week 2: Dimensionality Reduction - Exhaustive Deep Dive

## Independent Component Analysis (ICA)

### Theoretical Explanation

Independent Component Analysis (ICA) is a computational method for separating a multivariate signal into additive subcomponents. It does this by assuming that the subcomponents are non-Gaussian signals and that they are statistically independent from each other. ICA is a special case of blind source separation.

### Code Snippets

**Scikit-Learn:**

```python
from sklearn.decomposition import FastICA

# ICA
ica = FastICA(n_components=2)
S_ = ica.fit_transform(X)  # Reconstruct signals
A_ = ica.mixing_  # Get estimated mixing matrix
```

## Non-Negative Matrix Factorization (NMF)

### Theoretical Explanation

Non-Negative Matrix Factorization (NMF) is a dimensionality reduction technique that is used to factorize a non-negative matrix into two non-negative matrices. NMF is often used in text analysis and image processing, where the data is non-negative.

### Code Snippets

**Scikit-Learn:**

```python
from sklearn.decomposition import NMF

# NMF
nmf = NMF(n_components=2, init='random', random_state=0)
W = nmf.fit_transform(X)
H = nmf.components_
```

## Kernel PCA

### Theoretical Explanation

Kernel PCA is a non-linear dimensionality reduction technique that is an extension of PCA. It works by implicitly mapping the data into a higher-dimensional space where it is linearly separable, and then performing PCA in that space.

### Code Snippets

**Scikit-Learn:**

```python
from sklearn.decomposition import KernelPCA

# Kernel PCA
kernel_pca = KernelPCA(n_components=2, kernel='rbf', gamma=10)
X_kernel_pca = kernel_pca.fit_transform(X)
```

## Autoencoders for Dimensionality Reduction

### Theoretical Explanation

An autoencoder is a type of neural network that is used for unsupervised learning. It consists of two parts: an encoder and a decoder. The encoder maps the input data to a lower-dimensional representation, and the decoder maps the lower-dimensional representation back to the original data.

### Code Snippets

**TensorFlow/Keras:**

```python
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# Autoencoder
input_img = Input(shape=(784,))
encoded = Dense(32, activation='relu')(input_img)
decoded = Dense(784, activation='sigmoid')(encoded)

autoencoder = Model(input_img, decoded)
```

## Advanced Feature Selection

### Theoretical Explanation

*   **Recursive Feature Elimination with Cross-Validation (RFECV):** A feature selection technique that recursively removes features and then uses cross-validation to determine the optimal number of features.
*   **Stability Selection:** A feature selection technique that is based on resampling the data and then selecting the features that are most frequently selected.
*   **Mutual Information:** A measure of the mutual dependence between two variables. It can be used to select features that are most informative about the target variable.