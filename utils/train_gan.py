import os
import pandas as pd
from gan import GAN
from data_processing import process_data

def preprocess_data(df):
    # Perform any necessary preprocessing steps on the data
    return df

def train_gan(csv_files):
    num_features = 6
    num_historical_days = 5 * 252
    generator_input_size = 300
    epochs = 100

    gan = GAN(num_features, num_historical_days, generator_input_size)
    gan.discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    gan.combined.compile(loss='binary_crossentropy', optimizer='adam')

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        df[numeric_columns] = (df[numeric_columns] - df[numeric_columns].mean()) / df[numeric_columns].std()

        gan.train(df[numeric_columns].values, epochs)

    # Save the trained GAN model
    gan.generator.save('models/gan_generator')
    gan.discriminator.save('models/gan_discriminator')

# Specify the path to the CSV files
csv_folder = 'data'

# Get a list of all CSV files in the folder
csv_files = [os.path.join(csv_folder, file) for file in os.listdir(csv_folder) if file.endswith('.csv')]

# Train the GAN model using the CSV files
train_gan(csv_files)
