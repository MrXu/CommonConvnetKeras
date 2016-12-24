# https://github.com/rcmalli/keras-squeezenet
from common_convnet.utils.utils import plot_model_cwd
import os
from keras.layers import Input, merge, Flatten, Dense, AveragePooling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dropout, Activation
from keras.layers.pooling import GlobalAveragePooling2D
from keras.models import Model, Sequential

sq1x1 = "squeeze1x1"
exp1x1 = "expand1x1"
exp3x3 = "expand3x3"
relu = "relu_"

dir_path = os.path.dirname(os.path.realpath(__file__))
visual_path = os.path.join(dir_path, "visual")

# Modular function for Fire Node

def fire_module(x, fire_id, squeeze=16, expand=64, dim_ordering='tf'):
    s_id = 'fire' + str(fire_id) + '/'
    if dim_ordering is 'tf':
        c_axis = 3
    else:
        c_axis = 1

    x = Convolution2D(squeeze, 1, 1, border_mode='valid', name=s_id + sq1x1)(x)
    x = Activation('relu', name=s_id + relu + sq1x1)(x)

    left = Convolution2D(expand, 1, 1, border_mode='valid', name=s_id + exp1x1)(x)
    left = Activation('relu', name=s_id + relu + exp1x1)(left)

    right = Convolution2D(expand, 3, 3, border_mode='same', name=s_id + exp3x3)(x)
    right = Activation('relu', name=s_id + relu + exp3x3)(right)

    x = merge([left, right], mode='concat', concat_axis=c_axis, name=s_id + 'concat')
    return x


# Original SqueezeNet from paper.

def get_squeezenet(nb_classes, dim_ordering='tf'):
    if dim_ordering is 'th':
        input_img = Input(shape=(3, 227, 227))
    elif dim_ordering is 'tf':
        input_img = Input(shape=(227, 227, 3))
    else:
        raise NotImplementedError("Theano and Tensorflow are only available")
    x = Convolution2D(64, 3, 3, subsample=(2, 2), border_mode='valid', name='conv1')(input_img)
    x = Activation('relu', name='relu_conv1')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

    x = fire_module(x, fire_id=2, squeeze=16, expand=64, dim_ordering=dim_ordering)
    x = fire_module(x, fire_id=3, squeeze=16, expand=64, dim_ordering=dim_ordering)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)

    x = fire_module(x, fire_id=4, squeeze=32, expand=128, dim_ordering=dim_ordering)
    x = fire_module(x, fire_id=5, squeeze=32, expand=128, dim_ordering=dim_ordering)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)

    x = fire_module(x, fire_id=6, squeeze=48, expand=192, dim_ordering=dim_ordering)
    x = fire_module(x, fire_id=7, squeeze=48, expand=192, dim_ordering=dim_ordering)
    x = fire_module(x, fire_id=8, squeeze=64, expand=256, dim_ordering=dim_ordering)
    x = fire_module(x, fire_id=9, squeeze=64, expand=256, dim_ordering=dim_ordering)
    x = Dropout(0.5, name='drop9')(x)

    x = Convolution2D(nb_classes, 1, 1, border_mode='valid', name='conv10')(x)
    x = Activation('relu', name='relu_conv10')(x)
    x = GlobalAveragePooling2D()(x)
    out = Activation('softmax', name='loss')(x)
    model = Model(input=input_img, output=[out])

    # plot original squeezenet_v1.1
    plot_model_cwd(model, visual_path, "squeezenet_v1.png")

    return model


def get_squeezenet_pretrained(dim_ordering='tf'):
    model = get_squeezenet(1000, dim_ordering)

    weights_path = "common_convnet/squeezenet/squeezenet_weights_tf_dim_ordering_tf_kernels.h5"
    assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
    model.load_weights(weights_path)

    print "before pop layers: " + str(len(model.layers))

    model.layers.pop() # Get rid of the classification layer
    model.layers.pop() # Get rid of the average pooling layer
    model.layers.pop() # Get rid of the activation layer
    model.layers.pop() # Get rid of the convolution layer
    model.layers.pop() # Get rid of the dropout layer
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []

    print "after pop layers: " + str(len(model.layers))
    print model.outputs
    print model.output_shape
    print model.inbound_nodes
    print model.layers

    plot_model_cwd(model, visual_path, "squeezenet_v1_pretrained.png")

    return model


def get_squeezenet_top(input_shape, weights_file=None):
    model = Sequential()
    model.add(Convolution2D(2, 1, 1, border_mode='valid', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1, activation='sigmoid'))

    plot_model_cwd(model, visual_path, "squeezenet_top_original.png")

    if weights_file:
        model.load_weights(weights_file)

    return model

def get_squeezenet_top_softmax(input_shape, output_num, weight_file=None):
    model = Sequential()
    model.add(Convolution2D(2, 1, 1, border_mode='valid', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(output_dim=output_num, init="he_normal", activation="softmax"))

    plot_model_cwd(model, visual_path, "squeezenet_top_original_softmax.png")

    if weight_file:
        model.load_weights(weight_file)

    return model

def get_squeezenet_top_maxpool(input_shape, weights_file=None, pool_type="max"):
    model = Sequential()
    if pool_type=="max":
        model.add(MaxPooling2D(pool_size=(7,7), strides=None, input_shape=input_shape))
    elif pool_type=="ave":
        model.add(AveragePooling2D(pool_size=(7,7), strides=None, input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(1, activation='sigmoid'))

    if weights_file:
        model.load_weights(weights_file)

    return model

def get_squeezenet_top_maxpool_size(input_shape, pool_size=6, strides=None, weights_file=None):
    model = Sequential()
    model.add(MaxPooling2D(pool_size=(pool_size,pool_size), strides=strides, input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    plot_model_cwd(model, visual_path, "squeezenet_top_maxpool_{}.png".format(str(pool_size)))

    if weights_file:
        model.load_weights(weights_file)

    return model

def get_squeezenet_top_model_convnet(input_shape, weights_file=None):
    model = Sequential()
    model.add(MaxPooling2D(pool_size=(7,7), strides=(1,1), input_shape=input_shape))
    model.add(Convolution2D(nb_filter=64, nb_row=3, nb_col=3, border_mode='valid', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3,3), strides=(1,1), input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    plot_model_cwd(model, visual_path, "squeezenet_top_convnet.png")

    if weights_file:
        model.load_weights(weights_file)

    return model