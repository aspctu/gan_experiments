from keras.layers import Input, Dense, Activation, Flatten, Reshape
from keras.layers import UpSampling2D, Conv2D, Conv2DTranspose, LeakyReLU
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.optimizers import RMSprop
from keras.models import Model, Sequential
import discriminator as d

class Adverserial():
    def __init__(self, img_rows, img_cols, img_channels):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channels = img_channels
        self.d_model = None
        self.a_model = None
        self.g_model = None
        self.depth = 256
        self.dim = 7
        self.dropout = .4

    def generate_generator(self):
        '''
        Method used to generate adverserial network by importing Discrominator network, 
        writing Generator network and then stacking both networks
        '''
        generator = Sequential([
        Dense(128*7*7, input_dim=100, activation=LeakyReLU(0.2)),
        BatchNormalization(),
        Reshape((7,7,128)),
        UpSampling2D(),
        Conv2D(64, 5, 5, border_mode='same', activation=LeakyReLU(0.2)),
        BatchNormalization(),
        UpSampling2D(),
        Conv2D(1, 5, 5, border_mode='same', activation='tanh')
        ])
        self.g_model = generator
        self.g_model.compile(loss = 'binary_crossentropy', optimizer = RMSprop(lr = 0.0001, \
                decay = 3e-8), metrics = ['accuracy'])
        return(self.g_model)

    def generate_adverserial(self):
        '''
        Method used to stack generator and adverserial 
        '''
        self.g_model = self.generate_generator()
        print(self.g_model)
        discriminator = d.Discriminator(self.img_rows, self.img_cols, self.channels)
        discriminator.generate_discriminator()
        discriminator.compile_discriminator()
        self.d_model = discriminator.d_model
        self.d_model.trainable = False
        
        adverserial_x = Input(shape=(100,))
        x = self.g_model(adverserial_x)
        ganOutput = self.d_model(x)
        adverserial = Model(input=adverserial_x, output=ganOutput)
        self.a_model = adverserial
        
        
    def compile_adverserial(self):
        '''
        Method used to compile adverserial model (stacked generator and discriminator networks)
        '''
        self.a_model.compile(loss = 'binary_crossentropy', optimizer = RMSprop(lr = 0.0001, \
                decay = 3e-8), metrics = ['accuracy'])
        return(self.a_model)