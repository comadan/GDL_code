import os
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input, Conv2D, BatchNormalization, Dropout, Dense, Flatten, Reshape, Conv2DTranspose, LeakyReLU, Lambda, Activation
from keras.models import Model
from keras.backend import int_shape, mean, square, random_normal, exp, sum, shape
from keras.optimizers import Adam
from keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint
from keras.utils import plot_model


class VariationalAutoEncoder():
    """
    Classic autoencoder model with encoder and decoder architecture.
    Built for Generative Deep Learning textbook.
    """
    def __init__(self,
        input_dim,
        latent_dim,
        encoder_params,
        decoder_params,
        use_batch_norm=False,
        use_dropout=False,
        dropout_rate=.1,
        reconstruction_loss_multiplier=1e3
        ):
        """
        args:
            input_dim: tuple shape of the input data
            encoder_params: [{"filter_size": filter_size, "kernel_shape": kernel_shape, "strides": strides}, ... ]
            decoder_params: [{"filter_size": filter_size, "kernel_shape": kernel_shape, "strides": strides}, ... ]
            latent_dim: dimension of the compressed latent vector output by the encoder and input to the decoder.
        """
        
        self.name = "classicautoencoder"
        
        self.input_dim = input_dim
        self.encoder_params = encoder_params
        self.decoder_params = decoder_params
        self.latent_dim = latent_dim
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.reconstruction_loss_multiplier = reconstruction_loss_multiplier
        
        
        self._build()
    
    
    def _build(self):
        # encoder
        encoder_input = Input(shape=self.input_dim, name="encoder_input")
        
        layer = encoder_input
        
        for (i, p) in enumerate(self.encoder_params):
            convolutional_layer = Conv2D(
                filters=p['filter'],
                kernel_size=p['kernel'],
                strides=p['stride'],
                padding='same',
                name=f'encoder_conv_{i}')
            
            layer = convolutional_layer(layer)
            layer = LeakyReLU()(layer)
            
            if self.use_batch_norm:
                layer = BatchNormalization()(layer)
            
            if self.use_dropout:
                layer = Dropout(rate=self.dropout_rate)(layer)
        
        final_convolutional_shape = int_shape(layer)[1:] # this is used to map to a 2D space from the intial dense vectors in the decoder
        
        layer = Flatten()(layer)
        
        self.mu = Dense(self.latent_dim, name="mu")(layer)
        self.log_variance = Dense(self.latent_dim, name="log_variance")(layer)
        
        self.encoder_mu_log_variance = Model(encoder_input, (self.mu, self.log_variance))
        
        def sampling(args):
            mu, log_variance = args
            epsilon = random_normal(shape=shape(mu), mean=0., stddev=1.)
            return self.mu + epsilon * exp(self.log_variance / 2)
        
        encoder_output = Lambda(sampling, name='encoder_output')([self.mu, self.log_variance])
        
        self.encoder_model = Model(encoder_input, encoder_output)
        
        
        # decoder
        decoder_input = Input(shape=(self.latent_dim,), name="decoder_input")
        layer = decoder_input
        
        layer = Dense(np.prod(final_convolutional_shape))(layer)
        layer = Reshape(final_convolutional_shape)(layer)
        
        for (i, p) in enumerate(self.decoder_params):
            deconvolutional_layer = Conv2DTranspose(
                filters=p['filter'],
                kernel_size=p['kernel'],
                strides=p['stride'],
                padding='same',
                name=f'decoder_conv_{i}')
            
            layer = deconvolutional_layer(layer)
            
            if i < len(self.decoder_params) - 1:
                layer = LeakyReLU()(layer)
                
                if self.use_batch_norm:
                    layer = BatchNormalization()(layer)
                
                if self.use_dropout:
                    layer = Dropout(rate=self.dropout_rate)(layer)
            else:
                layer = Activation('sigmoid', name="decoder_output")(layer)
            
        decoder_output = layer
        
        self.decoder_model = Model(decoder_input, decoder_output)
        
        
        # full autoencoder model
        self.autoencoder_model = Model(encoder_input, self.decoder_model(encoder_output))
        
        # reconstruction loss
        def reconstruction_loss(y_pred, y_true):
            return mean(square(y_pred - y_true), axis=[1, 2, 3]) * self.reconstruction_loss_multiplier
        
        self.reconstruction_loss = reconstruction_loss
        
        def kl_divergence_loss(y_pred, y_true):
            return -.5 * sum(1 + self.log_variance - square(self.mu) - exp(self.log_variance), axis=1)
        
        self.kl_divergence_loss = kl_divergence_loss
        
        def vae_loss(y_pred, y_true):
            return self.reconstruction_loss(y_pred, y_true) + self.kl_divergence_loss(y_pred, y_true)
        
        self.vae_loss = vae_loss
    
    
    def compile(self, learning_rate):
        self.learning_rate = learning_rate
        optimizer = Adam(lr=self.learning_rate)
        
        self.autoencoder_model.compile(optimizer=optimizer, loss=self.vae_loss, metrics=[self.reconstruction_loss, self.kl_divergence_loss])
    
    
    def save(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
            os.makedirs(os.path.join(folder, 'viz'))
            os.makedirs(os.path.join(folder, 'weights'))
            os.makedirs(os.path.join(folder, 'images'))

        with open(os.path.join(folder, 'params.pkl'), 'wb') as f:
            pickle.dump([
                self.input_dim,
                self.latent_dim,
                self.encoder_params,
                self.decoder_params,
                self.use_batch_norm,
                self.use_dropout,
                self.dropout_rate,
                ], f)

        self.plot_model(folder)
        
        self.autoencoder_model.save_weights(os.path.join(folder, 'weights/weights.h5'))
    
    
    def load_weights(self, filepath):
        self.autoencoder_model.load_weights(filepath)
    
    
    def plot_model(self, run_folder):
        plot_model(self.autoencoder_model, to_file=os.path.join(run_folder, 'viz/autoencoder_model.png'), show_shapes=True, show_layer_names=True)
        plot_model(self.decoder_model, to_file=os.path.join(run_folder, 'viz/decoder_model.png'), show_shapes=True, show_layer_names=True)
        plot_model(self.encoder_model, to_file=os.path.join(run_folder, 'viz/encoder_model.png'), show_shapes=True, show_layer_names=True)
        
    
    def train(self, x_train, batch_size, epochs, run_folder, print_every_n_batches = 100, initial_epoch = 0, lr_decay = 1):
        def learning_rate_schedule(epoch):
            return self.learning_rate * (lr_decay ** epoch)
        
        self.learning_rate_scheduler = LearningRateScheduler(learning_rate_schedule) # The callbacks adjust the learning rate used by the optimization algorithm
        
        custom_callback = CustomCallback(run_folder, print_every_n_batches, initial_epoch, self)
        checkpoint = ModelCheckpoint(os.path.join(run_folder, 'weights/weights.h5'), save_weights_only = True, verbose=1)
        callbacks = [custom_callback, checkpoint, self.learning_rate_scheduler]
        
        self.autoencoder_model.fit(
            x_train,
            x_train,
            batch_size=batch_size,
            epochs=epochs,
            initial_epoch=initial_epoch,
            shuffle=True,
            callbacks=callbacks,
            )
        

def load_model(model_class, folder):
    
    with open(os.path.join(folder, 'params.pkl'), 'rb') as f:
        params = pickle.load(f)

    model = model_class(*params)

    model.load_weights(os.path.join(folder, 'weights/weights.h5'))

    return model


class CustomCallback(Callback):
    
    def __init__(self, run_folder, print_every_n_batches, initial_epoch, ae):
        self.epoch = initial_epoch
        self.run_folder = run_folder
        self.print_every_n_batches = print_every_n_batches
        self.ae = ae

    def on_batch_end(self, batch, logs={}):  
        if batch % self.print_every_n_batches == 0:
            latent_new = np.random.normal(size = (1,self.ae.latent_dim))
            reconst = self.ae.decoder_model.predict(np.array(latent_new))[0].squeeze()

            filepath = os.path.join(self.run_folder, 'images', 'img_' + str(self.epoch).zfill(3) + '_' + str(batch) + '.jpg')
            if len(reconst.shape) == 2:
                plt.imsave(filepath, reconst, cmap='gray_r')
            else:
                plt.imsave(filepath, reconst)

    def on_epoch_end(self, epoch, logs={}):
        self.epoch += 1
