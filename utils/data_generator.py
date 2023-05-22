import glob
import numpy as np
from data_processing import process_data
from gan import GAN

def train_gan(gan, generator, discriminator, data, epochs=2000, batch_size=32):
    for epoch in range(epochs):
        for company_data in data:
            # Generate fake data
            noise = np.random.normal(0, 1, [batch_size, 30])
            generated_data = generator.predict(noise)
            # Get real data
            real_data = get_real_data(company_data, batch_size)
            # Concatenate real and fake data, and create labels
            X = np.concatenate([real_data, generated_data])
            y = np.zeros(2 * batch_size)
            y[:batch_size] = 1  # Real data is 1, fake data is 0
            # Train discriminator
            discriminator.trainable = True
            discriminator.train_on_batch(X, y)
            # Train generator
            noise = np.random.normal(0, 1, [batch_size, 30])
            y_gen = np.ones(batch_size)
            discriminator.trainable = False
            gan.train_on_batch(noise, y_gen)

def get_real_data(data, batch_size):
    # Select random samples from the real data
    idx = np.random.randint(0, data.shape[0], batch_size)
    return data[idx]

def main():
    csv_files = glob.glob('*.csv')  # Replace this with the path to your CSV files
    data = process_data(csv_files)  # process_data is from data_processing.py
    gan = GAN(num_features=6, num_historical_days=50, generator_input_size=300)  # GAN is from gan.py
    train_gan(gan, gan.generator, gan.discriminator, data)

if __name__ == '__main__':
    main()
