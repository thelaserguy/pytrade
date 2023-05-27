from gan import GAN
from keras.models import load_model
import numpy as np

def generate_synthetic_data():
    num_features = 6  
    num_historical_days = 3 * 252
    generator_input_size = 300

    generator = load_model('models/gan_generator')
    discriminator = load_model('models/gan_discriminator')

    # Adjust the input shape of your generator, it should be 300, not 100
    noise_dim = generator_input_size  

    # Number of synthetic data samples you want to generate
    num_samples = 1000  

    # Create a noise array of size (num_samples, noise_dim)
    noise = np.random.normal(0, 1, size=(num_samples, noise_dim))

    # Now you can use this noise array to generate synthetic data
    synthetic_data = generator.predict(noise)
    
    # Save the synthetic data to a .npy file
    np.save('synthetic_data.npy', synthetic_data)
    
    print(synthetic_data)

if __name__ == "__main__":
    generate_synthetic_data()
