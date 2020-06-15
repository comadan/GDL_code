import os, pickle
from collections import deque

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
                 buffer_max_length=50):
        
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
        
        self.buffer_max_length = buffer_max_length
        self.buffer_A = deque(maxlen = self.buffer_max_length)
        self.buffer_B = deque(maxlen = self.buffer_max_length)
        
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
        
        output = Conv2D(self.input_dim[-1], kernel_size=4, strides=1, padding="same")(u_0)
        
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
        layer = disc_conv(input_layer, self.discriminator_first_layer_filters * 2, strides=2)
        layer = disc_conv(input_layer, self.discriminator_first_layer_filters * 4, strides=2)
        layer = disc_conv(input_layer, self.discriminator_first_layer_filters * 8, strides=2)
        
        
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
        

    def train_discriminators(self):
        return (
            d_loss_total[0]
            , dA_loss[0], dA_loss_real[0], dA_loss_fake[0]
            , dB_loss[0], dB_loss_real[0], dB_loss_fake[0]
            , d_loss_total[1]
            , dA_loss[1], dA_loss_real[1], dA_loss_fake[1]
            , dB_loss[1], dB_loss_real[1], dB_loss_fake[1]
        )

    def train_generators(self):
        
        return
    
    
    def train(self):
        
        return
    
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
                self.discriminator_loss,
                self.buffer_max_length,], f)

        self.plot_model(folder)


    def save_model(self, run_folder):
        self.adversarial_model.save(os.path.join(run_folder, 'adversarial_model.h5')  )
        self.discriminator_A.save(os.path.join(run_folder, 'discriminator_A.h5') )
        self.discriminator_B.save(os.path.join(run_folder, 'discriminator_B.h5') )
        self.translator_BA.save(os.path.join(run_folder, 'translator_BA.h5')  )
        self.translator_AB.save(os.path.join(run_folder, 'translator_AB.h5') )

        pickle.dump(self, open( os.path.join(run_folder, "model_obj.pkl"), "wb" ))
    
    
    def load_weights(self, filepath):
        self.combined.load_weights(filepath)

