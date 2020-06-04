
import numpy as np

from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, UpSampling2D, BatchNormalization, Dropout, Dense, Flatten, Reshape, Lambda, Activation
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam, RMSprop
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
                 generator_learning_rate=.0001,
                 discriminator_learning_rate=.0001,
                 generator_batch_norm_momentum=None,
                 discriminator_batch_norm_momentum=None,
                 generator_dropout_rate=None,
                 discriminator_dropout_rate=None,
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
        
        self.generator_batch_norm_momentum = generator_batch_norm_momentum
        self.discriminator_batch_norm_momentum = discriminator_batch_norm_momentum
        
        self.generator_dropout_rate = generator_dropout_rate
        self.discriminator_dropout_rate = discriminator_dropout_rate

        self.weight_initializer = RandomNormal(mean=0., stddev=.02)
        
        self.current_epoch = 0

        self._build_generator()
        self._build_discriminator()
        self._build_adversarial()


    def _build_generator(self):
        generator_input = Input(shape=(self.latent_dim, ), name="generator_input")

        layer = Dense(units=np.prod(self.generator_initial_dim), kernel_initializer=self.weight_initializer)(generator_input)
        if self.generator_batch_norm_momentum is not None:
            layer = BatchNormalization(momentum=self.generator_batch_norm_momentum)(layer)
        
        layer = Activation(self.generator_activation)(layer)
        layer = Reshape(target_shape=self.generator_initial_dim)(layer)
        if (self.generator_dropout_rate is not None) and (self.generator_dropout_rate > 0.):
            layer = Dropout(rate=self.generator_dropout_rate)(layer)

        # convolutional layers
        for i, param in enumerate(self.generator_convolutional_params):
            if ('upsample' in param) and (param['upsample'] is not None) and param['upsample'] > 1:
                layer = UpSampling2D(size=(param['upsample'], param['upsample']))(layer)

            layer = Conv2D(filters=param['filters'], kernel_size=param['kernel_size'], strides=param['strides'], padding='same', kernel_initializer=self.weight_initializer, name=f'generator_conv2d_{i}')(layer)

            if i < len(self.generator_convolutional_params) - 1:
                if self.generator_batch_norm_momentum is not None:
                    layer = BatchNormalization(momentum=self.generator_batch_norm_momentum)(layer)
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
            
            if (self.discriminator_batch_norm_momentum is not None) and (i < len(self.discriminator_convolutional_params) - 1) and (self.discriminator_dense_dim is not None) and (self.discriminator_dense_dim > 0):
                layer = BatchNormalization(self.discriminator_batch_norm_momentum)(layer)
            layer = Activation(self.discriminator_activation)(layer)
            if (self.discriminator_dropout_rate is not None) and (self.discriminator_dropout_rate > 0.):
                layer = Dropout(rate=self.discriminator_dropout_rate)(layer)
        
        layer = Flatten()(layer)
        if (self.discriminator_dense_dim is not None) and (self.discriminator_dense_dim > 0):
            layer = Dense(units=self.discriminator_dense_dim, kernel_initializer=self.weight_initializer)(layer)
            layer = Activation(self.discriminator_activation)(layer)
        
        layer = Dense(units=1, activation='sigmoid', kernel_initializer=self.weight_initializer)(layer)
        
        self.discriminator_model = Model(discriminator_input, layer, name="discriminator_model")
    
    
    def set_model_trainable(self, model, is_trainable=True):
        """
        sets the model and layers trainable parameter to arg: is_trainable {bool} 
        """
        model.trainable = is_trainable
        for layer in model.layers:
            layer.trainable = is_trainable
    
        
    def _build_adversarial(self):
        self.discriminator_model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=self.discriminator_learning_rate), metrics=['accuracy'])
        
        self.set_model_trainable(self.discriminator_model, is_trainable=False)
        adversarial_input = Input(shape=(self.latent_dim, ), name="adversarial_input")
        adversarial_output = self.discriminator_model(self.generator_model(adversarial_input))
        self.adversarial_model = Model(adversarial_input, adversarial_output, name="adversarial_model")
        
        self.adversarial_model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=self.generator_learning_rate), metrics=['accuracy'])
        self.set_model_trainable(self.discriminator_model, is_trainable=True)    
 
    
    def train_generator(self, batch_size):
        valid = np.ones((batch_size, 1))
        generated = np.zeros((batch_size, 1))
        latent_noise = np.random.normal(0., 1., (batch_size, self.latent_dim))
        # generated_images = self.generator_model.predict(latent_noise)
        # print("pre generator training", self.discriminator_model.test_on_batch(generated_images, valid), self.discriminator_model.test_on_batch(generated_images, generated))
        stats = self.adversarial_model.train_on_batch(latent_noise, valid)
        # generated_images = self.generator_model.predict(latent_noise)
        # print("post generator training", self.discriminator_model.test_on_batch(generated_images, valid), self.discriminator_model.test_on_batch(generated_images, generated))
        # print("post training", self.discriminator_model.test_on_batch(generated_images, valid), self.discriminator_model.test_on_batch(generated_images, generated))
        return stats
    
    
    def train_discriminator(self, x_train, batch_size):
        valid = np.ones((batch_size, 1))
        generated = np.zeros((batch_size, 1))
        y = np.concatenate((valid, generated), axis=0)
        
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        valid_images = x_train[idx]
        
        latent_noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
        generated_images = self.generator_model.predict(latent_noise)
        
        x = np.concatenate((valid_images, generated_images), axis=0)
        valid_stats, generated_stats = self.discriminator_model.test_on_batch(valid_images, valid), self.discriminator_model.test_on_batch(generated_images, generated)
        # print("pre discriminator training", valid_stats, generated_stats)
        self.discriminator_model.train_on_batch(x, y)
        # valid_stats, generated_stats = self.discriminator_model.test_on_batch(valid_images, valid), self.discriminator_model.test_on_batch(generated_images, generated)
        # print("post discriminator training", valid_stats, generated_stats)
        
        return valid_stats, generated_stats
        
    
    # def train_discriminator(self, x_train, batch_size):
    #     valid = np.ones((batch_size, 1))
    #     generated = np.zeros((batch_size, 1))
        
    #     idx = np.random.randint(0, x_train.shape[0], batch_size)
    #     valid_images = x_train[idx]
        
    #     latent_noise = np.random.normal(0., 1., (batch_size, self.latent_dim))
    #     generated_images = self.generator_model.predict(latent_noise)
    #     # print(self.discriminator_model.test_on_batch(generated_images, generated))
        
    #     v = self.discriminator_model.train_on_batch(valid_images, valid)
    #     g = self.discriminator_model.train_on_batch(generated_images, generated)
    #     return v, g
    
    
    def train(self, x_train, batch_size, epochs, run_folder, print_every_n_batches = 50):
        for epoch in range(self.current_epoch, self.current_epoch + epochs):
            valid_stats, generated_stats = self.train_discriminator(x_train, batch_size)
            generator_stats = self.train_generator(batch_size)
            print(f"epoch: {epoch}  disc. loss: (v: {valid_stats[0]:.3f} g: {generated_stats[0]:.3f}) acc.: (v: {valid_stats[1]:.3f} g: {generated_stats[1]:.3f})  gen. loss:{generator_stats[0]:.3f} acc.: {generator_stats[1]:.3f}")
            # if epoch % print_every_n_batches == 0:
            self.current_epoch += 1

