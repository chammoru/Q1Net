import tensorflow as tf
from tensorflow.python.keras.layers import ReLU, Conv2D, BatchNormalization, Dense, GlobalAveragePooling2D, \
    GlobalMaxPooling2D, Reshape, multiply, Concatenate, Add, Activation, Lambda
from tensorflow.python.keras import backend as K


def bottleneck(x, planes):
    out = C2Dx(x, planes, 1)
    out = BNx(out)
    out = ReLU()(out)

    out = C2DxP(out, planes, 3)
    out = BNx(out)
    out = ReLU()(out)

    out = C2Dx(out, planes, 1)
    out = BNx(out)

    out = tf.keras.layers.add([x, out])
    out = ReLU()(out)
    return out


def C2Dx(x, planes, kernel_size):
    """basic Conv2D block

    """

    return Conv2D(filters=planes, kernel_size=kernel_size, use_bias=False)(x)


def C2DxP(x, planes, kernel_size):
    """padding added

    """
    return Conv2D(filters=planes, kernel_size=kernel_size, padding='same', use_bias=False)(x)


def C2DxS(x, planes, kernel_size):
    """sigmoid activation

    """
    return Conv2D(filters=planes, kernel_size=kernel_size, activation='sigmoid', use_bias=True)(x)


def BNx(x):
    """BatchNormalization with initialization matching python

    """
    return BatchNormalization(momentum=0.9, epsilon=1e-05)(x)


def CBR(x, planes, kernel_size):
    x = C2Dx(x, planes, kernel_size)
    x = BNx(x)
    x = ReLU()(x)
    return x


def allow_gpu_growth():
    # Prevent the "CUDNN_STATUS_ALLOC_FAILED" error
    tf.keras.backend.clear_session()
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.compat.v1.InteractiveSession(config=config)


def to_tflite(model, tflite_model_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)
    print("Saved TensorFlow lite model to", tflite_model_path)


def cbam_block(input_feature, channel, ratio=2, kernel_size=7, kernel_initializer='he_normal'):
    """An implementation of Convolutional Block Attention Module(CBAM) block,
    as described in https://arxiv.org/abs/1807.06521.
    """

    cbam_feature = channel_attention(input_feature, channel, ratio, kernel_initializer)
    cbam_feature = spatial_attention(cbam_feature, kernel_size)
    return cbam_feature


def channel_attention(input_feature, channel, ratio=2, kernel_initializer='he_normal'):
    shared_layer_one = Dense(channel // ratio,
                             activation='relu',
                             kernel_initializer=kernel_initializer,
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer=kernel_initializer,
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)

    ca_feature = Add()([avg_pool, max_pool])
    ca_feature = Activation('sigmoid')(ca_feature)

    return multiply([input_feature, ca_feature])


def spatial_attention(input_feature, kernel_size=7):
    sa_feature = input_feature
    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(sa_feature)
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(sa_feature)
    concat = Concatenate(axis=3)([avg_pool, max_pool])

    sa_feature = Conv2D(filters=1,
                        kernel_size=kernel_size,
                        strides=1,
                        padding='same',
                        activation='sigmoid',
                        kernel_initializer='he_normal',
                        use_bias=False)(concat)

    return multiply([input_feature, sa_feature])
