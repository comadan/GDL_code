import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from functools import partial

from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, UpSampling2D, BatchNormalization, Dropout, Dense, Flatten, Reshape, Lambda, Activation, LeakyReLU
from keras.layers.merge import _Merge
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam, RMSprop
from keras.initializers import RandomNormal
from keras.utils import plot_model

import keras.backend as backend


class RandomWeightedAverage(_Merge):
    """
    keras needs each layer to be a layer type to build a model out of it.
    """
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        alpha = backend.random_uniform((self.batch_size, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

class WGANGP():
    """
    Wasserstein Generative Adversarial Network with Gradient Penalty
    """
    def __init__(self,
                 image_dim,
                 latent_dim,
                 generator_initial_dim,
                 critic_dense_dim, 
                 generator_activation,
                 critic_activation,
                 generator_convolutional_params,
                 critic_convolutional_params,
                 batch_size,
                 generator_learning_rate=.0001,
                 critic_learning_rate=.0001,
                 generator_batch_norm_momentum=None,
                 critic_batch_norm_momentum=None,
                 generator_dropout_rate=None,
                 critic_dropout_rate=None,
                 gradient_penalty_weight=10,
                 ):
        """
        args:
            generator_convolutional_params: [{'strides': {Int}, 'upsample': {int}, 'filter_size': {int}, 'kernel_size': (int, int)}, 
                                            ...]
        """
        self.image_dim = image_dim
        self.latent_dim = latent_dim
        
        self.generator_initial_dim = generator_initial_dim
        self.critic_dense_dim = critic_dense_dim

        self.generator_activation = generator_activation
        self.critic_activation = critic_activation

        self.generator_convolutional_params = generator_convolutional_params
        self.critic_convolutional_params = critic_convolutional_params
        
        self.generator_learning_rate = generator_learning_rate
        self.critic_learning_rate = critic_learning_rate
        
        self.generator_batch_norm_momentum = generator_batch_norm_momentum
        self.critic_batch_norm_momentum = critic_batch_norm_momentum
        
        self.generator_dropout_rate = generator_dropout_rate
        self.critic_dropout_rate = critic_dropout_rate

        self.weight_initializer = RandomNormal(mean=0., stddev=.02)
        
        self.batch_size = batch_size
        
        self.current_epoch = 0
        self.critic_losses = []
        self.generator_losses = []
        
        self.gradient_penalty_weight = gradient_penalty_weight

        self._build_generator()
        self.generator_model.summary()
        self._build_critic()
        self.critic_model.summary()
        self._build_adversarial()
        self.critic_gp_model.summary()
        self.adversarial_model.summary()
    
    
    def get_activation(self, activation):
        if activation == 'leaky_relu':
            layer = LeakyReLU(alpha = 0.2)
        else:
            layer = Activation(activation)
        return layer


    def _build_generator(self):
        generator_input = Input(shape=(self.latent_dim, ), name="generator_input")

        layer = Dense(units=np.prod(self.generator_initial_dim), kernel_initializer=self.weight_initializer)(generator_input)
        if self.generator_batch_norm_momentum is not None:
            layer = BatchNormalization(momentum=self.generator_batch_norm_momentum)(layer)
        
        layer = self.get_activation(self.generator_activation)(layer)
        layer = Reshape(target_shape=self.generator_initial_dim)(layer)
        if (self.generator_dropout_rate is not None) and (self.generator_dropout_rate > 0.):
            layer = Dropout(rate=self.generator_dropout_rate)(layer)

        # convolutional layers
        for i, param in enumerate(self.generator_convolutional_params):
            if ('upsample' in param) and (param['upsample'] is not None) and param['upsample'] > 1:
                layer = UpSampling2D(size=(param['upsample'], param['upsample']))(layer)
            
            if ('transpose' in param) and param['transpose']:
                layer = Conv2DTranspose(filters=param['filters'], kernel_size=param['kernel_size'], strides=param['strides'], padding='same', kernel_initializer=self.weight_initializer, name=f'generator_conv2dtranspose_{i}')(layer)
            else:
                layer = Conv2D(filters=param['filters'], kernel_size=param['kernel_size'], strides=param['strides'], padding='same', kernel_initializer=self.weight_initializer, name=f'generator_conv2d_{i}')(layer)

            if i < len(self.generator_convolutional_params) - 1:
                if self.generator_batch_norm_momentum is not None:
                    layer = BatchNormalization(momentum=self.generator_batch_norm_momentum)(layer)
                layer = self.get_activation(self.generator_activation)(layer)
                if (self.generator_dropout_rate is not None) and (self.generator_dropout_rate > 0.):
                    layer = Dropout(rate=self.generator_dropout_rate)(layer)
            else:
                layer = Activation('tanh', name='generator_final_activation')(layer)

        self.generator_model = Model(generator_input, layer, name="generator_model")
    
        
    def _build_critic(self):
        critic_input = Input(shape=self.image_dim, name="critic_input")
        layer = critic_input
        for i, param in enumerate(self.critic_convolutional_params):
            layer = Conv2D(filters=param['filters'], kernel_size=param['kernel_size'], strides=param['strides'], padding='same', kernel_initializer=self.weight_initializer, name=f'critic_conv2d_{i}')(layer)
            
            if (self.critic_batch_norm_momentum is not None) and ((i < len(self.critic_convolutional_params) - 1) or ((self.critic_dense_dim is not None) and (self.critic_dense_dim > 0))):
                layer = BatchNormalization(momentum=self.critic_batch_norm_momentum)(layer)
            layer = self.get_activation(self.critic_activation)(layer)
            if (self.critic_dropout_rate is not None) and (self.critic_dropout_rate > 0.):
                layer = Dropout(rate=self.critic_dropout_rate)(layer)
        
        layer = Flatten()(layer)
        if (self.critic_dense_dim is not None) and (self.critic_dense_dim > 0):
            layer = Dense(units=self.critic_dense_dim, kernel_initializer=self.weight_initializer)(layer)
            layer = self.get_activation(self.critic_activation)(layer)
        
        layer = Dense(units=1, activation=None, kernel_initializer=self.weight_initializer)(layer)
        
        self.critic_model = Model(critic_input, layer, name="critic_model")
        
    
    def set_model_trainable(self, model, is_trainable=True):
        """
        sets the model and layers trainable parameter to arg: is_trainable {bool} 
        """
        model.trainable = is_trainable
        for layer in model.layers:
            layer.trainable = is_trainable
    
        
    def _build_adversarial(self):
        ################################################
        # construct computational graph for the critic #
        ################################################
        self.set_model_trainable(self.generator_model, False)
        real_images = Input(shape=self.image_dim)
        
        generator_latent_dim = Input(shape=(self.latent_dim, ))
        generated_images = self.generator_model(generator_latent_dim)
        
        real_scores = self.critic_model(real_images)
        generated_scores = self.critic_model(generated_images)
        
        
        interpolated_images = RandomWeightedAverage(self.batch_size)([real_images, generated_images])
        
        interpolated_scores = self.critic_model(interpolated_images)
        
        partial_gradient_penalty = partial(self.gradient_penalty_cost, interpolated_images=interpolated_images)
        partial_gradient_penalty.__name__ = 'gradient_penalty' # name required by keras
        
        # although the inerpolated scores aren't really used in the loss, we output them here, because we need 3 outputs for the loss function.
        self.critic_gp_model = Model(inputs=[real_images, generator_latent_dim], outputs=[real_scores, generated_scores, interpolated_scores])
        self.critic_gp_model.compile(loss=[self.wasserstein_cost, self.wasserstein_cost, partial_gradient_penalty], 
            optimizer=Adam(lr=self.critic_learning_rate, beta_1=0.5),
            loss_weights=[1, 1, self.gradient_penalty_weight])
        
        ###################################################
        # construct computational graph for the generator #
        ###################################################
        self.set_model_trainable(self.critic_model, is_trainable=False)
        self.set_model_trainable(self.generator_model, is_trainable=True)
        adversarial_input = Input(shape=(self.latent_dim, ), name="adversarial_input")
        adversarial_output = self.critic_model(self.generator_model(adversarial_input))
        self.adversarial_model = Model(adversarial_input, adversarial_output, name="adversarial_model")
        
        self.adversarial_model.compile(loss=self.wasserstein_cost, optimizer=Adam(lr=self.generator_learning_rate, beta_1=0.5), metrics=['accuracy'])
        self.set_model_trainable(self.critic_model, is_trainable=True)    
    
    
    def wasserstein_cost(self, y_true, y_pred):
        return - backend.mean(y_true * y_pred)
    
    
    def gradient_penalty_cost(self, y_true, y_pred, interpolated_images):
        '''
        compute lambda * (1 - ||grad||)^2 still for each single sample
        '''
        
        gradients = backend.gradients(y_pred, interpolated_images)[0]
        
        gradient_l2_norm = backend.sqrt(backend.sum(backend.square(gradients), axis=np.arange(1, len(gradients.shape))))
        
        gradient_penalty = backend.square(1 - gradient_l2_norm)
        
        return backend.mean(gradient_penalty)
    
    def train_generator(self, batch_size):
        valid = np.ones((batch_size, 1))
        latent_noise = np.random.normal(0., 1., (batch_size, self.latent_dim))
        stats = self.adversarial_model.train_on_batch(latent_noise, valid)
        return stats
    
    
    def train_critic(self, x_train, batch_size, clip_threshold):
        valid = np.ones((batch_size, 1))
        generated = -np.ones((batch_size, 1))
        dummy = np.zeros((batch_size, 1))
                
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        valid_images = x_train[idx]
        
        latent_noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
        generated_images = self.generator_model.predict(latent_noise)
        
        # valid_stats, generated_stats = self.critic_model.test_on_batch(valid_images, valid), self.critic_model.test_on_batch(generated_images, generated)
        loss = self.critic_gp_model.train_on_batch([valid_images, latent_noise], [valid, generated, dummy])
                
        return loss
        
    
    def train(self, x_train, batch_size, epochs, run_folder, print_every_n_batches=50, critic_training_steps=5, clip_threshold = 0.01):
        paths = [os.path.join(run_folder, subdir) for subdir in ["weights", "model", "sampled_images"]]
        for p in paths:
            if not os.path.exists(p):
                os.makedirs(p)
        
        for epoch in range(self.current_epoch, self.current_epoch + epochs):
            for _ in range(critic_training_steps):
                critic_loss = self.train_critic(x_train, batch_size, clip_threshold)
            
            generator_stats = self.train_generator(batch_size)
            print(f"epoch: {epoch}  disc. loss: {critic_loss[0]:.3f} (v: {critic_loss[1]:.3f} g: {critic_loss[2]:.3f} gp: {critic_loss[3]:.3f}) gen. loss:{generator_stats[0]:.3f} acc.: {generator_stats[1]:.3f}")
            
            self.generator_losses.append(generator_stats)
            self.critic_losses.append(critic_loss)

            if epoch % print_every_n_batches == 0:
                self.sample_images(run_folder)
                self.adversarial_model.save_weights(os.path.join(run_folder, f"weights/weights-{self.current_epoch}.h5"))
            self.current_epoch += 1
        
        self.adversarial_model.save_weights(os.path.join(run_folder, f"weights/weights.h5"))
        self.save_model(run_folder)
    
    
    def sample_images(self, run_folder):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator_model.predict(noise)

        gen_imgs = 0.5 * (gen_imgs + 1)
        gen_imgs = np.clip(gen_imgs, 0, 1)

        fig, axs = plt.subplots(r, c, figsize=(15,15))
        cnt = 0

        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(np.squeeze(gen_imgs[cnt, :,:,:]), cmap = 'gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig(os.path.join(run_folder, "sampled_images/sample_%d.png" % self.current_epoch))
        plt.close()
    
    
    def save_model(self, run_folder):
        self.adversarial_model.save(os.path.join(run_folder, "model/adversarial_model.h5"))
        self.generator_model.save(os.path.join(run_folder, "model/generator_model.h5"))
        self.critic_model.save(os.path.join(run_folder, "model/critic_model.h5"))
        pickle.dump(self, open(os.path.join(run_folder, "model/gan_object.pkl"), "wb"))
    
    
    def load_model(self, filepath):
        self.adversarial_model.load_weights(filepath)
    
    
    def plot_model(self, run_folder):
        plot_model(self.adversarial_model, to_file=os.path.join(run_folder ,'viz/model.png'), show_shapes = True, show_layer_names = True)
        plot_model(self.critic_model, to_file=os.path.join(run_folder ,'viz/critic.png'), show_shapes = True, show_layer_names = True)
        plot_model(self.generator_model, to_file=os.path.join(run_folder ,'viz/generator.png'), show_shapes = True, show_layer_names = True)
    
    
    def save(self, folder):

        with open(os.path.join(folder, 'params.pkl'), 'wb') as f:
            pickle.dump([
                 self.image_dim,
                 self.latent_dim,
                 self.generator_initial_dim,
                 self.critic_dense_dim, 
                 self.generator_activation,
                 self.critic_activation,
                 self.generator_convolutional_params,
                 self.critic_convolutional_params,
                 self.generator_learning_rate,
                 self.critic_learning_rate,
                 self.generator_batch_norm_momentum,
                 self.critic_batch_norm_momentum,
                 self.generator_dropout_rate,
                 self.critic_dropout_rate,
                ], f)

        self.plot_model(folder)

        


