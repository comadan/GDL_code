from keras.layers import Conv2D, Activation, UpSampling2D, Concatenate
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

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
                 discriminator_loss=None):
        
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
        
        return

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
        
        input_layer = Input(self.image_dim)
        
        # downsample
        d_1 = downsample(input_layer, self.translator_first_layer_filters)
        d_2 = downsample(input_layer, self.translator_first_layer_filters * 2)
        d_3 = downsample(input_layer, self.translator_first_layer_filters * 4)
        d_4 = downsample(input_layer, self.translator_first_layer_filters * 8)
        
        # upsample
        u_3 = upsample(d_4, d_3, self.translator_first_layer_filters * 4)
        u_2 = upsample(u_3, d_2, self.translator_first_layer_filters * 2)
        u_1 = upsample(u_2, d_1, self.translator_first_layer_filters * 1)
        
        u_0 = UpSampling2D(size=(2, 2))(u_1)
        
        return Model(input_layer, u_0)


    def build_translator_resnet(self):
        return
    

    def build_discriminator(self):
        def disc_conv(layer_input, filters, strides=2, instance_normalization=True):
            layer = Conv2D(filters, kernel_size=(4, 4), strides=strides, padding='same')(layer_input)
            if instance_normalization:
                layer = InstanceNormalization(axis=-1, center=False, scale=False)(layer)
            layer = LeakyReLu(0.2)(layer)
        
        input_layer = Input(shape=self.image_dim)
        
        layer = disc_conv(input_layer, self.discriminator_first_layer_filters, stride=2, norm=False) # first layer doesn't use instance normalization
        layer = disc_conv(input_layer, self.discriminator_first_layer_filters * 2, stride=2)
        layer = disc_conv(input_layer, self.discriminator_first_layer_filters * 4, stride=2)
        layer = disc_conv(input_layer, self.discriminator_first_layer_filters * 8, stride=2)
        
        
        if (self.discriminator_loss == 'binary_crossentropy'):
            output_layer = Conv2D(1, kernel_size=(4, 4), strides=1, padding='same', activation='sigmoid')(layer)
        elif (self.discriminator_loss == 'mse'):
            output_layer = Conv2D(1, kernel_size=(4, 4), strides=1, padding='same')(layer) # output layer of PatchGAN with no activation function
        
        return Model(input_layer, output_layer)
    
    def set_model_trainable(self, model, trainable):
        model.trainable = trainable
        for l in model.layers:
            layer.trainable = trainable
    
    def compile_models(self):
        self.discriminator_A = self.build_discriminator()
        self.discriminator_B = self.build_discriminator()
        
        self.discriminator_A.compile(loss=self.discriminator_loss, optimizer=Adam(self.learning_rate, beta_1=0.5), metrics=['accuracy'])
        self.discriminator_B.compile(loss=self.discriminator_loss, optimizer=Adam(self.learning_rate, beta_1=0.5), metrics=['accuracy'])
        
        # set discriminators to be not trainable for compiling translators in GAN
        self.set_model_trainable(self.discriminator_A, False)
        self.set_model_trainable(self.discriminator_B, False)
        
        
        if self.translator_model_type == "unet":
            self.translator_BA = build_translator_unet()
            self.translator_AB = build_translator_unet()
        elif self.translator_model_type == "resnet":
            self.translator_BA =build_translator_resnet()
            self.translator_AB =build_translator_resnet()
        
        
        image_A = Input(shape=self.image_dim)
        image_B = Input(shape=self.image_dim)
        
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
                                       loss_weights=[self.lambda_discriminator, self.lambda_discriminator,
                                                     self.lambda_reconstruction, self.lambda_reconstruction,
                                                     self.lambda_identity, self.lambda_identity])
        

    def compile(self):
        return

