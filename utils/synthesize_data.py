from gan import GAN
from keras.models import load_model
import numpy as np

def generate_synthetic_data():
    num_features = 6  
    num_historical_days = 50
    generator_input_size = 300

    generator = load_model('models/gan_generator')
    discriminator = load_model('models/gan_discriminator')

    # Assuming the input shape of your generator takes a vector of size 100
    noise_dim = 100  

    # Number of synthetic data samples you want to generate
    num_samples = 1000  

    # Create a noise array of size (num_samples, noise_dim)
    noise = np.random.normal(0, 1, size=(num_samples, noise_dim))

    # Now you can use this noise array to generate synthetic data
    synthetic_data = generator.predict(noise)
    print(synthetic_data)

if __name__ == "__main__":
    generate_synthetic_data()
