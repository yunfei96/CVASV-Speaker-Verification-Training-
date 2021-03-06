import tensorflow as tf



def get_conv_block(tensor, channels, strides, alpha=1.0, name=''):
    channels = int(channels * alpha)

    x = tf.keras.layers.Conv2D(channels,
               kernel_size=(3, 3),
               strides=strides,
               use_bias=False,
               padding='same',
               name='{}_conv'.format(name))(tensor)
    x = tf.keras.layers.BatchNormalization(name='{}_bn'.format(name))(x)
    x = tf.keras.layers.Activation('relu', name='{}_act'.format(name))(x)
    return x


def get_dw_sep_block(tensor, channels, strides, alpha=1.0, name=''):
    """Depthwise separable conv: A Depthwise conv followed by a Pointwise conv."""
    channels = int(channels * alpha)

    # Depthwise
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3),
                        strides=strides,
                        use_bias=False,
                        padding='same',
                        name='{}_dw'.format(name))(tensor)
    x = tf.keras.layers.BatchNormalization(name='{}_bn1'.format(name))(x)
    x = tf.keras.layers.Activation('relu', name='{}_act1'.format(name))(x)

    # Pointwise
    x = tf.keras.layers.Conv2D(channels,
               kernel_size=(1, 1),
               strides=(1, 1),
               use_bias=False,
               padding='same',
               name='{}_pw'.format(name))(x)
    x = tf.keras.layers.BatchNormalization(name='{}_bn2'.format(name))(x)
    x = tf.keras.layers.Activation('relu', name='{}_act2'.format(name))(x)
    return x


def MobileNet(shape, num_classes, alpha=1.0, include_top=True, weights=None):
    x_in = tf.keras.layers.Input(shape=shape)

    x = get_conv_block(x_in, 32, (2, 2), alpha=alpha, name='initial')

    layers = [
        (64, (1, 1)),
        (128, (2, 2)),
        (128, (1, 1)),
        (256, (2, 2)),
        (256, (1, 1)),
        (512, (2, 2)),
        *[(512, (1, 1)) for _ in range(5)],
        (1024, (2, 2)),
        (1024, (2, 2))
    ]

    for i, (channels, strides) in enumerate(layers):
        x = get_dw_sep_block(x, channels, strides, alpha=alpha, name='block{}'.format(i))

    if include_top:
        x = tf.keras.layers.GlobalAveragePooling2D(name='gavg')(x)
        x = tf.keras.layers.Dense(num_classes, activation='softmax', name='softmax')(x)

    model = tf.keras.Model(inputs=x_in, outputs=x)

    if weights is not None:
        model.load_weights(weights, by_name=True)

    return model