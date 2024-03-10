import numpy as np
import os
import struct
import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from train_NN import train_CNN

def Main():
    data_dir = 'Data/MNIST/raw'

    with open(os.path.join(data_dir, 'train-images-idx3-ubyte'), 'rb') as f:
            magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
            images = np.fromfile(f, dtype=np.uint8).reshape(num_images, rows*cols)

    with open(os.path.join(data_dir, 'train-labels-idx1-ubyte'), 'rb') as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)


    image = images[34].reshape(rows, cols)
    N = images.shape[0]
    ### Adding a channel dimension, since train_NN is intended 
    ### to be generalized to 3-color
    images = images.reshape(N, 1, 28, 28)
    print(f"Shape of an image is {image.shape}")
    print(f"Pixels range from {np.min(image)} to {np.max(image)}")

    ### ~~~Constructing data ~~~
    train_split = 0.85 
    ### Hyperparams
    batch_size = 2
    learning_rate = 0.01
    validation_split = 0.85
    epochs = 15
    hyperparams = (batch_size, learning_rate, validation_split, epochs)
    
    print(f"Shape of the full structure is {images.shape}")
    norm_img = images / 255.0
    shuffled_indices = np.random.permutation(norm_img.shape[0])

    norm_img = norm_img[shuffled_indices]
    labels = labels[shuffled_indices]

    train_size = int(train_split * norm_img.shape[0])
    train_set = norm_img[:train_size]
    test_set = norm_img[train_size:]

    train_labels = labels[:train_size]
    test_labels = labels[train_size:]



    ### Expected image shape: N x C x W x H 
    ### N = number of samples
    ### C = channel number (1 for MNIST)
    ### W, H self explanatory
    model = train_CNN(train_set, train_labels, hyperparams)


if __name__ == '__main__':
    Main()

