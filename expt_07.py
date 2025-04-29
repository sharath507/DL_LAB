import numpy as np, tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Load and normalize data
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train.astype("float32") / 255., x_test.astype("float32") / 255.
x_train, x_test = x_train[..., np.newaxis], x_test[..., np.newaxis]

# Add noise function
def add_noise(x): return np.clip(x + 0.5 * np.random.randn(*x.shape), 0., 1.)
x_train_noisy, x_test_noisy = add_noise(x_train), add_noise(x_test)

# Undercomplete Autoencoder
enc = models.Sequential([
    layers.Input((28, 28, 1)),
    layers.Flatten(),
    layers.Dense(64, activation='relu')
])

dec = models.Sequential([
    layers.Input((64,)),
    layers.Dense(784, activation='sigmoid'),
    layers.Reshape((28, 28, 1))
])

ucae = models.Sequential([enc, dec])
ucae.compile(optimizer='adam', loss='binary_crossentropy')
ucae.fit(x_train, x_train, epochs=5, batch_size=356, verbose=1)

# Denoising Autoencoder
dnae = models.Sequential([enc, dec])  # same architecture
dnae.compile(optimizer='adam', loss='binary_crossentropy')
dnae.fit(x_train_noisy, x_train, epochs=5, batch_size=356, verbose=1)

# VAE Sampling Layer
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# VAE Encoder
inputs = layers.Input(shape=(28, 28, 1))
x = layers.Flatten()(inputs)
x = layers.Dense(128, activation='relu')(x)
z_mean = layers.Dense(2, name='z_mean')(x)
z_log_var = layers.Dense(2, name='z_log_var')(x)
z = Sampling()([z_mean, z_log_var])
encoder = tf.keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")

# VAE Decoder
latent_inputs = layers.Input(shape=(2,))
x = layers.Dense(128, activation='relu')(latent_inputs)
x = layers.Dense(784, activation='sigmoid')(x)
outputs = layers.Reshape((28, 28, 1))(x)
decoder = tf.keras.Model(latent_inputs, outputs, name="decoder")

# VAE Model
class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        recon_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(inputs, reconstructed)) * 784
        kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        self.add_loss(recon_loss + kl_loss)
        return reconstructed

vae = VAE(encoder, decoder)
vae.compile(optimizer='adam')
vae.fit(x_train, epochs=5, batch_size=356, verbose=1)

# Show Images
def show_imgs(orig, rec, title="", n=10):
    plt.figure(figsize=(20, 4))
    n = min(n, 25)
    for i in range(n):
        # Original
        plt.subplot(2, n, i + 1)
        img = orig[i].squeeze() if orig[i].ndim == 3 else orig[i]
        plt.imshow(img, cmap='gray')
        plt.axis('off')

        # Reconstructed
        plt.subplot(2, n, i + 1 + n)
        img = rec[i].squeeze() if rec[i].ndim == 3 else rec[i]
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# Show results
show_imgs(x_test, ucae.predict(x_test), "Undercomplete Autoencoder")
show_imgs(x_test, dnae.predict(x_test_noisy), "Denoising Autoencoder")
show_imgs(x_test, vae.predict(x_test), "Variational Autoencoder")
