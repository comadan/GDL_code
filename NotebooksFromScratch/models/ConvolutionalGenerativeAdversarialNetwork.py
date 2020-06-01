import numpy as np

from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, UpSampling2D, BatchNormalization, Dropout, Dense, Flatten, Reshape, Lambda, Activation
from keras.initializers import RandomNormal

import keras.backend as backend

class ConvolutionalGenerativeAdversarialNetwork():
    def __init__(self,
                 image_dim,
                 latent_dim,
                 generator_initial_dim,
                 generator_activation,
                 discriminator_activation,
                 generator_convolutional_params,
                 discriminator_convolutional_params,
                 use_batch_norm=False,
                 use_dropout=False,
                 dropout_rate=.1,
                 ):
        """
        args:
            generator_convolutional_params: [{'strides': {Int}, 'upsample': {int}, 'filter_size': {int}, 'kernel_size': (int, int)}, 
                                            ...]
        """
        self.generator_initial_dim = generator_initial_dim
        self.image_dim = image_dim
        self.latent_dim = latent_dim

        self.generator_activation = generator_activation

        self.generator_convolutional_params = generator_convolutional_params
        self.discriminator_convolutional_params = discriminator_convolutional_params

        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate

        self.weight_initializer = RandomNormal(mean=0., stddev=.02)

        self._build_generator()
        self._build_discriminator()
        self._compile_models()


    def _build_generator(self):
        generator_input = Input(shape=self.latent_dim, name="generator_input")

        layer = Dense(units=np.prod(self.generator_initial_dim))(generator_input)
        layer = Reshape(target_shape=self.generator_initial_dim)(layer)
        if self.use_batch_norm:
            layer = BatchNormalization()(layer)
        layer = Activation(self.generator_activation)(layer)

        # convolutional layers
        for i, param in enumerate(self.generator_convolutional_params):
            if ('upsample' in param) and (param['upsample'] is not None) and param['upsample'] > 1:
                layer = UpSampling2D(size=(param['upsample'], param['upsample']))(layer)

            layer = Conv2D(filters=param['filters'], kernel_size=param['kernel_size'], strides=param['strides'], padding='same', name=f'generator_conv2d_{i}')(layer)

            if i < len(self.generator_convolutional_params) - 1:
                layer = Activation(self.generator_activation)(layer)
                if self.use_batch_norm:
                    layer = BatchNormalization()(layer)
            else:
                layer = Activation('tanh', name='generator_final_activation')(layer)

        self.generator_model = Model(generator_input, layer)


    def _build_discriminator(self):
        return

    def _compile_models(self):
        return

    def train_discriminator(self):
        return

    def train_generator(self):
        return
