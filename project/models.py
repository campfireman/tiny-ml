from tensorflow.keras.layers import (
    Activation,
    Add,
    AveragePooling1D,
    BatchNormalization,
    Conv1D,
    Conv2D,
    Dense,
    DepthwiseConv1D,
    Dropout,
    Flatten,
    Input,
    MaxPooling1D,
    ReLU,
    SeparableConv1D,
)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import l2


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

    x = SeparableConv1D(8, 3, activation='relu', padding='same')(inp)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)

    x = bottleneck_block(x, filters=16, dilation_rate=2)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)

    x = residual_block(x, filters=16)
    x = MaxPooling1D(2)(x)

    x = Flatten()(x)
    x = Dense(16, activation='relu',
              kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation='softmax', name='y_pred')(x)

    return Model(inputs=inp, outputs=out)


def get_convolutional_model(input_shape, num_classes):
    return Sequential([
        Input(input_shape),
        Conv1D(16, kernel_size=5, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),

        SeparableConv1D(36, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),

        Dropout(0.15),

        SeparableConv1D(36, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),

        Flatten(),

        Dense(32, activation='relu',
              kernel_regularizer=l2(1e-4)),
        Dropout(0.3),

        Dense(num_classes, activation='softmax', name='y_pred')
    ])


def ds_conv_block(x, filters):
    x = DepthwiseConv1D(
        kernel_size=3,
        padding='same',
    )(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = SeparableConv1D(
        filters, kernel_size=1, padding='same',
        depthwise_regularizer=l2(1e-4),
    )(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling1D()(x)
    x = Dropout(0.15)(x)
    return x


def get_ds_cnn_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)   # e.g. (351, 1)

    # First separable‐conv block
    x = SeparableConv1D(
        32, kernel_size=5, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling1D(pool_size=2)(x)

    # Additional separable‐conv blocks
    for filters in (64, 64, 64):
        x = ds_conv_block(x, filters)

    # Classifier head
    x = Flatten()(x)
    x = Dense(32, activation='relu',
              kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.25)(x)
    outputs = Dense(num_classes, activation='softmax', name='y_pred')(x)

    return Model(inputs, outputs)
