import numpy as np

from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, UpSampling2D, BatchNormalization, Dropout, Dense, Flatten, Reshape, Lambda, Activation
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam
from keras.initializers import RandomNormal

import keras.backend as backend

class GenerativeAdversarialNetwork():
    def __init__(self,
                 image_dim,
                 latent_dim,
                 generator_initial_dim,
                 discriminator_dense_dim, 
                 generator_activation,
                 discriminator_activation,
                 generator_convolutional_params,
                 discriminator_convolutional_params,
                 generator_learning_rate=.01,
                 discriminator_learning_rate=.01,
                 use_batch_norm=False,
                 generator_dropout_rate=None,
                 discriminator_dropout_rate=None
                 ):
        """
        args:
            generator_convolutional_params: [{'strides': {Int}, 'upsample': {int}, 'filter_size': {int}, 'kernel_size': (int, int)}, 
                                            ...]
        """
        self.image_dim = image_dim
        self.latent_dim = latent_dim
        
        self.generator_initial_dim = generator_initial_dim
        self.discriminator_dense_dim = discriminator_dense_dim

        self.generator_activation = generator_activation
        self.discriminator_activation = discriminator_activation

        self.generator_convolutional_params = generator_convolutional_params
        self.discriminator_convolutional_params = discriminator_convolutional_params
        
        self.generator_learning_rate = generator_learning_rate
        self.discriminator_learning_rate = discriminator_learning_rate
        
        self.use_batch_norm = use_batch_norm
        self.generator_dropout_rate = generator_dropout_rate
        self.discriminator_dropout_rate = discriminator_dropout_rate

        self.weight_initializer = RandomNormal(mean=0., stddev=.02)

        self._build_generator()
        self._build_discriminator()
        self._build_adversarial()
        self._compile_models()


    def _build_generator(self):
        generator_input = Input(shape=self.latent_dim, name="generator_input")

        layer = Dense(units=np.prod(self.generator_initial_dim), kernel_initializer=self.weight_initializer)(generator_input)
        layer = Reshape(target_shape=self.generator_initial_dim)(layer)
        # note that I moved the batch normalization here after the reshape rather than before the reshape (as in the master branch)
        # by keras default the mean is taken for axis=-1, collapsing the mean across all other axis.
        if self.use_batch_norm:
            layer = BatchNormalization()(layer)
        layer = Activation(self.generator_activation)(layer)
        if (self.generator_dropout_rate is not None) and (self.generator_dropout_rate > 0.):
            layer = Dropout(rate=self.generator_dropout_rate)(layer)

        # convolutional layers
        for i, param in enumerate(self.generator_convolutional_params):
            if ('upsample' in param) and (param['upsample'] is not None) and param['upsample'] > 1:
                layer = UpSampling2D(size=(param['upsample'], param['upsample']))(layer)

            layer = Conv2D(filters=param['filters'], kernel_size=param['kernel_size'], strides=param['strides'], padding='same', kernel_initializer=self.weight_initializer, name=f'generator_conv2d_{i}')(layer)

            if i < len(self.generator_convolutional_params) - 1:
                if self.use_batch_norm:
                    layer = BatchNormalization()(layer)
                layer = Activation(self.generator_activation)(layer)
                if (self.generator_dropout_rate is not None) and (self.generator_dropout_rate > 0.):
                    layer = Dropout(rate=self.generator_dropout_rate)(layer)
            else:
                layer = Activation('tanh', name='generator_final_activation')(layer)

        self.generator_model = Model(generator_input, layer, name="generator_model")


    def _build_discriminator(self):
        discriminator_input = Input(shape=self.image_dim, name="discriminator_input")
        layer = discriminator_input
        for i, param in enumerate(self.discriminator_convolutional_params):
            layer = Conv2D(filters=param['filters'], kernel_size=param['kernel_size'], strides=param['strides'], padding='same', kernel_initializer=self.weight_initializer, name=f'discriminator_conv2d_{i}')(layer)
            
            if (self.use_batch_norm) and (i < len(self.discriminator_convolutional_params) - 1) and (self.discriminator_dense_dim is not None) and (self.discriminator_dense_dim > 0):
                layer = BatchNormalization()(layer)
            layer = Activation(self.discriminator_activation)(layer)
            if (self.discriminator_dropout_rate is not None) and (self.discriminator_dropout_rate > 0.):
                layer = Dropout(rate=self.discriminator_dropout_rate)(layer)
        
        layer = Flatten()(layer)
        if (self.discriminator_dense_dim is not None) and (self.discriminator_dense_dim > 0):
            layer = Dense(units=self.discriminator_dense_dim, kernel_initializer=self.weight_initializer)(layer)
            layer = Activation(self.discriminator_activation)(layer)
        
        layer = Dense(units=1, activation='sigmoid', kernel_initializer=self.weight_initializer)(layer)
        
        self.discriminator_model = Model(discriminator_input, layer, name="discriminator_model")
        
        BinaryCrossentropy()
    
    def _build_adversarial(self):
        adversarial_input = Input(shape=self.latent_dim, name="adversarial_input")
        adversarial_output = self.discriminator_model(self.generator_model(adversarial_input))
        self.adversarial_model = Model(adversarial_input, adversarial_output, name="adversarial_model")
    
    def _compile_models(self):
        # compile the discriminator first
        discriminator_optimizer = Adam(lr=self.discriminator_learning_rate)
        self.discriminator_model.compile(loss=BinaryCrossentropy(), optimizer=discriminator_optimizer, metrics=['accuracy'])
        
        self.set_model_trainable(self.discriminator_model, is_trainable=False)
        generator_optimizer = Adam(lr=self.generator_learning_rate)
        self.adversarial_model.compile(loss=BinaryCrossentropy(), optimizer=generator_optimizer, metrics=['accuracy'])
        return
    
    
    def set_model_trainable(self, model, is_trainable=True):
        model.trainable = is_trainable
        for layer in model.layers:
            layer.trainable = False
    
    def train_discriminator(self):
        return

    def train_generator(self):
        return
