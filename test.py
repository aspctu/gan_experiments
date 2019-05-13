#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: abqader
"""

from keras.datasets import mnist 
from tqdm import tqdm
import numpy as np
import adverserial as a

# Data loading 
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_train = X_train.astype('float32')
X_train = (X_train - 127.5) / 127.5

# Generate network componenets and set up training 
adverserial_net = a.Adverserial(28,28,1)
adverserial_net.generate_adverserial()
adverserial_net.compile_adverserial()

adverserial_model = adverserial_net.a_model
discriminator_model = adverserial_net.d_model
generator_model = adverserial_net.g_model

# Training hyperparameters 
epoch=10
batch_size=128
batch_count = X_train.shape[0] // batch_size

#Begin training 
for i in range(epoch):
    for j in tqdm(range(batch_count)):
        # Input for the generator
        noise_input = np.random.rand(batch_size, 100)
        
        # getting random images from X_train of size=batch_size 
        # these are the real images that will be fed to the discriminator
        image_batch = X_train[np.random.randint(0, X_train.shape[0], size=batch_size)]
        
        # these are the predicted images from the generator
        predictions = generator_model.predict(noise_input, batch_size=batch_size)
        
        # the discriminator takes in the real images and the generated images
        X = np.concatenate([predictions, image_batch])
        
        # labels for the discriminator
        y_discriminator = [0]*batch_size + [1]*batch_size
        
        # Let's train the discriminator
        discriminator_model.trainable = True
        discriminator_model.train_on_batch(X, y_discriminator)
        
        # Let's train the generator
        noise_input = np.random.rand(batch_size, 100)
        y_generator = [1]*batch_size
        discriminator_model.trainable = False
        adverserial_model.train_on_batch(noise_input, y_generator)
        
        