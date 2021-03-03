import tensorflow as tf

from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from tensorflow.keras.layers import Dropout, Concatenate

INPUT_SHAPE = (256, 256, 3)

class TableNet:
    @staticmethod
    def build_table_decoder(inputs, pool3, pool4):
        x = Conv2D(512, (1, 1), activation='relu', name='conv7_table')(inputs)
        x = UpSampling2D(size=(2, 2))(x)

        x = Concatenate()([x, pool4])

        x = UpSampling2D(size=(2, 2))(x)

        x = Concatenate()([x, pool3])

        x = UpSampling2D(size=(2, 2))(x)
        x = UpSampling2D(size=(2, 2))(x)

        last = tf.keras.layers.Conv2DTranspose(
            3, 3, strides=2,
            padding='same', name='table_output')

        x = last(x)

        return x

    @staticmethod
    def build_column_decoder(inputs, pool3, pool4):
        x = Conv2D(512, (1, 1), activation='relu', name='block7_conv1_column')(inputs)
        x = Dropout(0.8, name='block7_dropout_column')(x)

        x = Conv2D(512, (1, 1), activation='relu', name='block8_conv1_column')(x)
        x = UpSampling2D(size=(2, 2))(x)

        x = Concatenate()([x, pool4])

        x = UpSampling2D(size=(2, 2))(x)

        x = Concatenate()([x, pool3])

        x = UpSampling2D(size=(2, 2))(x)
        x = UpSampling2D(size=(2, 2))(x)

        last = tf.keras.layers.Conv2DTranspose(
            3, 3, strides=2,
            padding='same', name='column_output')

        x = last(x)

        return x

    @staticmethod
    def vgg_base(inputs):
        base_model = tf.keras.applications.vgg19.VGG19(
            input_shape=INPUT_SHAPE,
            include_top=False,
            weights='imagenet')

        layer_names = ['block3_pool', 'block4_pool', 'block5_pool']
        layers = [base_model.get_layer(name).output for name in layer_names]

        pool_layers_model = tf.keras.Model(inputs=base_model.input, outputs=layers, name='VGG-19')
        pool_layers_model.trainable = False

        return pool_layers_model(inputs)

    @staticmethod
    def build():
        inputs = tf.keras.Input(shape=INPUT_SHAPE, name='input')

        pool_layers = TableNet.vgg_base(inputs)

        x = Conv2D(512, (1, 1), activation='relu', name='block6_conv1')(pool_layers[2])
        x = Dropout(0.8, name='block6_dropout1')(x)
        x = Conv2D(512, (1, 1), activation='relu', name='block6_conv2')(x)
        x = Dropout(0.8, name='block6_dropout2')(x)

        table_mask = TableNet.build_table_decoder(x, pool_layers[0], pool_layers[1])
        column_mask = TableNet.build_column_decoder(x, pool_layers[0], pool_layers[1])

        model = tf.keras.Model(inputs=inputs, outputs=[table_mask, column_mask], name='tablenet')
        table_model = tf.keras.Model(inputs=inputs, outputs=table_mask, name='tablenet_table')
        column_model = tf.keras.Model(inputs=inputs, outputs=column_mask, name='tablenet_column')

        return model, table_model, column_model

