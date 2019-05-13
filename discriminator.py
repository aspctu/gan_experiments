from keras.layers import Input, Dense, Activation, Flatten, Reshape
from keras.layers import UpSampling2D, Conv2D, Conv2DTranspose, MaxPooling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import RMSprop, Adam
from keras.models import Model, Sequential

class Discriminator():
    def __init__(self, img_rows, img_cols, channels):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channels = channels
        self.d_model = None
        self.depth = 64
        self.dropout = .4

    def generate_discriminator(self):
        '''
        Method used to generate discrimnator network based on img dimensions 
        '''
        discriminator = Sequential([
        Conv2D(64, 5, 5, subsample=(2,2), input_shape=(28,28,1), border_mode='same', activation=LeakyReLU(0.2)),
        Dropout(0.3),
        Conv2D(128, 5, 5, subsample=(2,2), border_mode='same', activation=LeakyReLU(0.2)),
        Dropout(0.3),
        Flatten(),
        Dense(1, activation='sigmoid')
        ])
        self.d_model = discriminator
        return(self.d_model)

    def compile_discriminator(self):
        '''
        Method used to compile generator before training
        '''
        self.d_model.compile(loss = 'binary_crossentropy', optimizer = RMSprop(lr = 0.0001, \
                decay = 3e-8), metrics = ['accuracy'])
        return(self.d_model)



    
