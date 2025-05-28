import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout, BatchNormalization, ReLU, Add, Input, SeparableConv1D


def bottleneck_block(x, filters, dilation_rate):
    y = Conv1D(filters//4, kernel_size=1, activation='relu')(x)
    y = Conv1D(filters//4, kernel_size=5, dilation_rate=dilation_rate,
               padding='same', activation='relu')(y)
    y = Conv1D(filters, kernel_size=1, activation='relu')(y)
    return y


def residual_block(x, filters):
    shortcut = x
    y = Conv1D(filters,  kernel_size=3, padding='same')(x)
    y = BatchNormalization()(y)
    y = ReLU()(y)
    y = Conv1D(filters, kernel_size=3, padding='same')(y)
    y = BatchNormalization()(y)
    out = Add()([shortcut, y])
    return ReLU()(out)


def get_residual_model(input_shape, num_classes):
    inp = Input(shape=input_shape)

    x = SeparableConv1D(8, 5, activation='relu', padding='same')(inp)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)

    x = bottleneck_block(x, filters=16, dilation_rate=2)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)

    x = residual_block(x, filters=16)
    x = MaxPooling1D(2)(x)

    x = Flatten()(x)
    x = Dense(16, activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation='softmax', name='y_pred')(x)

    return Model(inputs=inp, outputs=out)


def get_convolutional_model(input_shape, num_classes):
    return Sequential([
        Conv1D(16, kernel_size=5, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),

        Conv1D(32, kernel_size=5, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),

        Flatten(),

        Dense(32, activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        Dropout(0.3),

        Dense(num_classes, activation='softmax', name='y_pred')
    ])
