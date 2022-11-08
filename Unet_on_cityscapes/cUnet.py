import tensorflow as tf


def downsample(filters, size, apply_norm=False, apply_pool=True):
    """Downsamples an input.

    MaxPool2D => Conv2D => Conv2D => Batchnorm(optional)

    """
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()

    if apply_pool:
        result.add(tf.keras.layers.MaxPool2D())

    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=1, padding='same',
                               activation='relu',
                               kernel_initializer=initializer, use_bias=False))
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=1, padding='same',
                               activation='relu',
                               kernel_initializer=initializer, use_bias=False))

    if apply_norm:
        result.add(tf.keras.layers.BatchNormalization())

    return result


def upsample(filters, size, apply_norm=False, apply_dropout=False):
    """Upsamples an input.

    Conv2DTranspose => Conv2D => Conv2D => Batchnorm => Dropout

    """

    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=1, padding='same',
                               activation='relu',
                               kernel_initializer=initializer, use_bias=False))
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=1, padding='same',
                               activation='relu',
                               kernel_initializer=initializer, use_bias=False))

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    if apply_norm:
        result.add(tf.keras.layers.BatchNormalization())

    return result
