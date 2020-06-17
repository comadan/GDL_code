import os, pickle
import datetime
from collections import deque

import random
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import Input, Conv2D, Activation, UpSampling2D, Concatenate, LeakyReLU
from keras.initializers import RandomNormal
from keras.optimizers import Adam
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.utils import plot_model


class CycleGenerativeAdversarialNetwork():
    """
    translator==generator
    I named the 'generator' components as 'translator' components, 
    because it's really more of a translation model than a generation model.
    
    
    args:
        translator_first_layer_filters: dimension of the first layer
    
    """
    def __init__(self,
                 input_dim,
                 learning_rate,
                 lambda_discriminator,
                 lambda_reconstruction,
                 lambda_identity,
                 translator_model_type,
                 translator_first_layer_filters,
                 discriminator_first_layer_filters,
                 discriminator_loss=None,
                 ):
        
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        
        # cost multipliers/lambdas
        self.lambda_discriminator = lambda_discriminator
        self.lambda_reconstruction = lambda_reconstruction
        self.lambda_identity = lambda_identity
        
        self.translator_model_type = translator_model_type
        
        # number of filters in first layer
        self.translator_first_layer_filters = translator_first_layer_filters
        self.discriminator_first_layer_filters = discriminator_first_layer_filters
        
        # check valid losses
        if (discriminator_loss not in ['binary_cross_entropy', 'mse', 'rmse']):
            raise ValueError(f'{discriminator_loss} is not a valid discriminator loss')
        self.discriminator_loss = discriminator_loss
        
        self.discriminator_losses = []
        self.translator_losses = []
        self.epoch = 0
                
        # Calculate output shape of discriminator (PatchGAN)
        patch_dim = int(self.input_dim[0] / 2**3)
        self.discriminator_patch_dim = (patch_dim, patch_dim, 1)

        self.weight_init = RandomNormal(mean=0., stddev=0.02)

        self.compile_models()

    def build_translator_unet(self):
        """
        The unet translator:
        first downsamples, increasing filter size but decreasing spatial dimension, 
        then upsamples, decreasing feature size and increasing spatial dimension.
        """
        def downsample(layer_input, filters, kernel_size=4):
            layer = Conv2D(filters, kernel_size=kernel_size, strides=2, padding='same')(layer_input)
            layer = InstanceNormalization(axis=-1, center=False, scale=False)(layer)
            layer = Activation('relu')(layer)
            return layer
            
        def upsample(layer_input, skip_input, filters, kernel_size=4, dropout_rate=0.):
            """
            skip_input is attached to the output of the upsampled convolution
            """
            layer = UpSampling2D(size=(2, 2))(layer_input)
            layer = Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same')(layer)
            layer = InstanceNormalization(axis=-1, center=False, scale=False)(layer)
            layer = Activation('relu')(layer)
            if (dropout_rate is not None) and (dropout_rate):
                layer = Dropout(dropout_rate)(layer)
            layer = Concatenate(axis=-1)([layer, skip_input])
            return layer
        
        input_layer = Input(self.input_dim)
        
        # downsample
        d_1 = downsample(input_layer, self.translator_first_layer_filters)
        d_2 = downsample(d_1, self.translator_first_layer_filters * 2)
        d_3 = downsample(d_2, self.translator_first_layer_filters * 4)
        d_4 = downsample(d_3, self.translator_first_layer_filters * 8)
        
        # upsample
        u_3 = upsample(d_4, d_3, self.translator_first_layer_filters * 4)
        u_2 = upsample(u_3, d_2, self.translator_first_layer_filters * 2)
        u_1 = upsample(u_2, d_1, self.translator_first_layer_filters * 1)
        
        u_0 = UpSampling2D(size=(2, 2))(u_1)
        
        output = Conv2D(self.input_dim[-1], kernel_size=4, strides=1, padding="same", activation='tanh')(u_0)
        
        return Model(input_layer, output)


    def build_translator_resnet(self):
        return
    

    def build_discriminator(self):
        def disc_conv(layer_input, filters, strides=2, instance_normalization=True):
            layer = Conv2D(filters, kernel_size=(4, 4), strides=strides, padding='same')(layer_input)
            if instance_normalization:
                layer = InstanceNormalization(axis=-1, center=False, scale=False)(layer)
            layer = LeakyReLU(0.2)(layer)
            return layer
        
        input_layer = Input(shape=self.input_dim)
        
        layer = disc_conv(input_layer, self.discriminator_first_layer_filters, strides=2, instance_normalization=False) # first layer doesn't use instance normalization
        layer = disc_conv(layer, self.discriminator_first_layer_filters * 2, strides=2)
        layer = disc_conv(layer, self.discriminator_first_layer_filters * 4, strides=2)
        layer = disc_conv(layer, self.discriminator_first_layer_filters * 8, strides=1) # kind of interesting that book didn't downsample in final convolution, use strides=2, here. There's a discrepancy in the text, that it says we're predicting an 8x8 patch, but in the code we're actually doing 16x16 patch.
        
        
        if (self.discriminator_loss == 'binary_crossentropy'):
            output_layer = Conv2D(1, kernel_size=(4, 4), strides=1, padding='same', activation='sigmoid')(layer)
        elif (self.discriminator_loss == 'mse'):
            output_layer = Conv2D(1, kernel_size=(4, 4), strides=1, padding='same')(layer) # output layer of PatchGAN with no activation function
        
        return Model(input_layer, output_layer)
    
    def set_model_trainable(self, model, trainable):
        model.trainable = trainable
        for l in model.layers:
            l.trainable = trainable
    
    def compile_models(self):
        self.discriminator_A = self.build_discriminator()
        self.discriminator_B = self.build_discriminator()
        
        self.discriminator_A.compile(loss=self.discriminator_loss, optimizer=Adam(self.learning_rate, beta_1=0.5), metrics=['accuracy'])
        self.discriminator_B.compile(loss=self.discriminator_loss, optimizer=Adam(self.learning_rate, beta_1=0.5), metrics=['accuracy'])
        
        # set discriminators to be not trainable for compiling translators in GAN
        self.set_model_trainable(self.discriminator_A, False)
        self.set_model_trainable(self.discriminator_B, False)
        
        
        if self.translator_model_type == "unet":
            self.translator_BA = self.build_translator_unet()
            self.translator_AB = self.build_translator_unet()
        elif self.translator_model_type == "resnet":
            self.translator_BA = self.build_translator_resnet()
            self.translator_AB = self.build_translator_resnet()
        
        
        image_A = Input(shape=self.input_dim)
        image_B = Input(shape=self.input_dim)
        
        translated_A = self.translator_BA(image_B)
        translated_B = self.translator_AB(image_A)
        
        reconstructed_A = self.translator_BA(translated_B)
        reconstructed_B = self.translator_AB(translated_A)
        
        # if the translators are applied to the original image style, do they remain the same?
        identity_A = self.translator_BA(image_A)
        identity_B = self.translator_AB(image_B)
        
        score_translated_A = self.discriminator_A(translated_A)
        score_translated_B = self.discriminator_B(translated_B)
        
        self.adversarial_model = Model(inputs=[image_A, image_B],
                                       outputs=[score_translated_A, score_translated_B,
                                                reconstructed_A, reconstructed_B,
                                                identity_A, identity_B])
        
        self.adversarial_model.compile(loss=[self.discriminator_loss, self.discriminator_loss,
                                             'mae', 'mae',
                                             'mae', 'mae'],
                                       optimizer=Adam(self.learning_rate, beta_1=0.5),
                                       loss_weights=[self.lambda_discriminator, self.lambda_discriminator,
                                                     self.lambda_reconstruction, self.lambda_reconstruction,
                                                     self.lambda_identity, self.lambda_identity])

        self.set_model_trainable(self.discriminator_A, True)
        self.set_model_trainable(self.discriminator_B, True)
        

    def train_discriminators(self, images_A, images_B, y_real, y_translated, alternating_discriminator=True):
        
        translated_A = self.translator_BA.predict(images_B)
        translated_B = self.translator_AB.predict(images_A)
                
        if alternating_discriminator:
            d_A_loss_real = self.discriminator_A.train_on_batch(images_A, y_real)
            d_A_loss_translated = self.discriminator_A.train_on_batch(translated_A, y_translated)
            d_B_loss_real = self.discriminator_B.train_on_batch(images_B, y_real)
            d_B_loss_translated = self.discriminator_B.train_on_batch(translated_B, y_translated)
            d_A_loss = .5 * np.add(d_A_loss_real, d_A_loss_translated)
            d_B_loss = .5 * np.add(d_B_loss_real, d_B_loss_translated)
        else:
            d_A_loss_real = self.discriminator_A.test_on_batch(images_A, y_real)
            d_A_loss_translated = self.discriminator_A.test_on_batch(translated_A, y_translated)
            d_B_loss_real = self.discriminator_A.test_on_batch(images_B, y_real)
            d_B_loss_translated = self.discriminator_A.test_on_batch(translated_B, y_translated)
            d_A_loss = self.discriminator_A.train_on_batch(np.concatenate((images_A, translated_A), axis=0), np.concatenate((y_real, y_translated), axis=0))
            d_B_loss = self.discriminator_B.train_on_batch(np.concatenate((images_B, translated_B), axis=0), np.concatenate((y_real, y_translated), axis=0))
        
        d_loss_total = 0.5 * np.add(d_A_loss, d_B_loss)
        
        return (
            d_loss_total[0]
            , d_A_loss[0], d_A_loss_real[0], d_A_loss_translated[0]
            , d_B_loss[0], d_B_loss_real[0], d_B_loss_translated[0]
            , d_loss_total[1]
            , d_A_loss[1], d_A_loss_real[1], d_A_loss_translated[1]
            , d_B_loss[1], d_B_loss_real[1], d_B_loss_translated[1]
        )

    def train_translators(self, images_A, images_B, y_real):

        # Train the translators
        return self.adversarial_model.train_on_batch([images_A, images_B],
                                                     [y_real, y_real,
                                                     images_A, images_B,
                                                     images_A, images_B])
    
    
    def train(self, data_loader, run_folder, epochs, test_A_file, test_B_file, batch_size=1, sample_interval=100, alternating_discriminator=True):
        start_time = datetime.datetime.now()
        
        # ground truths
        y_real = np.ones((batch_size, ) + self.discriminator_patch_dim)
        y_translated = np.ones((batch_size, ) + self.discriminator_patch_dim)
        
        for epoch in range(self.epoch, epochs):
            for b, (images_A, images_B) in enumerate(data_loader.load_batch(batch_size=batch_size)):
                d_loss = self.train_discriminators(images_A, images_B, y_real, y_translated, alternating_discriminator=alternating_discriminator)
                g_loss = self.train_translators(images_A, images_B, y_real)
                
                elapsed_time = datetime.datetime.now() - start_time
                                
                print (f"[Epoch {self.epoch}/{epochs}] [Batch {b}/{data_loader.n_batches}] [D loss: {d_loss[0]:.3f}, acc: {100*d_loss[7]:.0f}] [G loss: {g_loss[0]:.3f}, adv: {np.sum(g_loss[1:3]):.3f}, recon: {np.sum(g_loss[3:5]):.3f}, id: {np.sum(g_loss[5:7]):.3f}] time: {elapsed_time}")
                self.discriminator_losses.append(d_loss)
                self.translator_losses.append(g_loss)
                
                # If at save interval => save generated image samples
                if b % sample_interval == 0:
                    self.sample_images(data_loader, b, run_folder, test_A_file, test_B_file)
                    self.adversarial_model.save_weights(os.path.join(run_folder, 'weights/weights-%d.h5' % (self.epoch)))
                    self.adversarial_model.save_weights(os.path.join(run_folder, 'weights/weights.h5'))
                    self.save_model(run_folder)

                
            self.epoch += 1
    
    def sample_images(self, data_loader, batch_i, run_folder, test_A_file, test_B_file):
        
        r, c = 2, 4

        for p in range(2):

            if p == 1:
                images_A = data_loader.load_data(domain="A", batch_size=1, is_testing=True)
                images_B = data_loader.load_data(domain="B", batch_size=1, is_testing=True)
            else:
                images_A = data_loader.load_img('data/%s/testA/%s' % (data_loader.dataset_name, test_A_file))
                images_B = data_loader.load_img('data/%s/testB/%s' % (data_loader.dataset_name, test_B_file))

            # Translate images to the other domain
            translated_B = self.translator_AB.predict(images_A)
            translated_A = self.translator_BA.predict(images_B)
            # Translate back to original domain
            reconstructed_A = self.translator_BA.predict(translated_B)
            reconstructed_B = self.translator_AB.predict(translated_A)

            # ID the images
            identity_A = self.translator_BA.predict(images_A)
            identity_B = self.translator_AB.predict(images_B)

            gen_imgs = np.concatenate([images_A, translated_B, reconstructed_A, identity_A, images_B, translated_A, reconstructed_B, identity_B])

            # Rescale images 0 - 1
            gen_imgs = 0.5 * gen_imgs + 0.5
            gen_imgs = np.clip(gen_imgs, 0, 1)

            titles = ['Original', 'Translated', 'Reconstructed', 'ID']
            fig, axs = plt.subplots(r, c, figsize=(25,12.5))
            cnt = 0
            for i in range(r):
                for j in range(c):
                    axs[i,j].imshow(gen_imgs[cnt])
                    axs[i, j].set_title(titles[j])
                    axs[i,j].axis('off')
                    cnt += 1
            fig.savefig(os.path.join(run_folder ,"images/%d_%d_%d.png" % (p, self.epoch, batch_i)))
            plt.close()


    def plot_model(self, run_folder):
        plot_model(self.adversarial_model, to_file=os.path.join(run_folder ,'viz/adversarial_model.png'), show_shapes = True, show_layer_names = True)
        plot_model(self.discriminator_A, to_file=os.path.join(run_folder ,'viz/discriminator_A.png'), show_shapes = True, show_layer_names = True)
        plot_model(self.discriminator_B, to_file=os.path.join(run_folder ,'viz/discriminator_B.png'), show_shapes = True, show_layer_names = True)
        plot_model(self.translator_BA, to_file=os.path.join(run_folder ,'viz/translator_BA.png'), show_shapes = True, show_layer_names = True)
        plot_model(self.translator_AB, to_file=os.path.join(run_folder ,'viz/translator_AB.png'), show_shapes = True, show_layer_names = True)


    def save(self, folder):
        with open(os.path.join(folder, 'params.pkl'), 'wb') as f:
            pickle.dump([
                self.input_dim,
                self.learning_rate,
                self.lambda_discriminator,
                self.lambda_reconstruction,
                self.lambda_identity,
                self.translator_model_type,
                self.translator_first_layer_filters,
                self.discriminator_first_layer_filters,
                self.discriminator_loss,], f)

        self.plot_model(folder)
        self.save_model(folder)


    def save_model(self, run_folder):
        self.adversarial_model.save(os.path.join(run_folder, 'weights/adversarial_model.h5')  )
        self.discriminator_A.save(os.path.join(run_folder, 'weights/discriminator_A.h5') )
        self.discriminator_B.save(os.path.join(run_folder, 'weights/discriminator_B.h5') )
        self.translator_BA.save(os.path.join(run_folder, 'weights/translator_BA.h5')  )
        self.translator_AB.save(os.path.join(run_folder, 'weights/translator_AB.h5') )

        pickle.dump(self, open( os.path.join(run_folder, "weights/model_obj.pkl"), "wb" ))
    
    
    def load_weights(self, filepath):
        self.combined.load_weights(filepath)

