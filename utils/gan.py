import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

class GAN(keras.Model):
    def __init__(self, num_features, num_historical_days, generator_input_size=300, is_train=True):
        super().__init__()
        self.latent_dim = generator_input_size
        self.generator = self.make_generator_model(generator_input_size, num_features, num_historical_days)
        self.discriminator = self.make_discriminator_model(num_features, num_historical_days)
        
        # Create the combined model here
        self.combined = keras.Model(self.generator.input, self.discriminator(self.generator.output))
        self.combined.compile(optimizer='adam', loss='binary_crossentropy')
        
    def load_model(self, generator_path, discriminator_path):
        self.generator = keras.models.load_model(generator_path)
        self.discriminator = keras.models.load_model(discriminator_path)

    def make_generator_model(self, generator_input_size, num_features, num_historical_days):
        model = tf.keras.Sequential()
        model.add(layers.Dense(generator_input_size, use_bias=True, input_shape=(generator_input_size,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Dense(num_historical_days * num_features))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Reshape((num_historical_days, num_features)))
        assert model.output_shape == (None, num_historical_days, num_features)
        
        return model

    def make_discriminator_model(self, num_features, num_historical_days):
        model = tf.keras.Sequential()
        model.add(layers.Flatten())
        model.add(layers.Dense(512))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Dense(256))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(1, activation='sigmoid'))
        
        return model

    def call(self, x):
        generated_data = self.generator(x)
        return self.discriminator(generated_data)

    def train(self, data, epochs, batch_size=128):
        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            # Select a random batch of data
            idx = np.random.randint(0, data.shape[0], batch_size)
            real_data = data[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a batch of new data
            gen_data = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(real_data, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_data, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, valid)

            # If at save interval => save generated image samples and plot the progress
            if epoch % 1000 == 0:
                print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

    def save_model(self, file_path):
        self.generator.save_weights(f'{file_path}_generator.h5')
        self.discriminator.save_weights(f'{file_path}_discriminator.h5')
