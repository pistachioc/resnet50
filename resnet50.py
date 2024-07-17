from keras.layers import Input, Dense, Add, Activation, BatchNormalization, Flatten, Conv2D, AveragePooling2D, \
    MaxPooling2D
from base_resNet import bottelneck_residual_block
from keras.models import Model, load_model


def ResNet50(input_shape, classes):
    """
    Model ResNet50 voi 5 stage gom 50 layer conv"""

    X_input = Input(input_shape)

    # Stage 1
    X = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2))(X_input)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = bottelneck_residual_block(X, 3, [64, 64, 256], reduce=True, s=1)
    X = bottelneck_residual_block(X, 3, [64, 64, 256], reduce=False, s=2)
    X = bottelneck_residual_block(X, 3, [64, 64, 256], reduce=False, s=2)

    # Stage 3
    X = bottelneck_residual_block(X, 3, [128, 128, 512], reduce=True, s=2)
    X = bottelneck_residual_block(X, 3, [128, 128, 512], reduce=False, s=2)
    X = bottelneck_residual_block(X, 3, [128, 128, 512], reduce=False, s=2)
    X = bottelneck_residual_block(X, 3, [128, 128, 512], reduce=False, s=2)

    # Stage 4
    X = bottelneck_residual_block(X, 3, [256, 256, 1024], reduce=True, s=2)
    X = bottelneck_residual_block(X, 3, [256, 256, 1024], reduce=False, s=2)
    X = bottelneck_residual_block(X, 3, [256, 256, 1024], reduce=False, s=2)
    X = bottelneck_residual_block(X, 3, [256, 256, 1024], reduce=False, s=2)
    X = bottelneck_residual_block(X, 3, [256, 256, 1024], reduce=False, s=2)
    X = bottelneck_residual_block(X, 3, [256, 256, 1024], reduce=False, s=2)

    # Stage 5
    X = bottelneck_residual_block(X, 3, [512, 512, 2048], reduce=True, s=2)
    X = bottelneck_residual_block(X, 3, [512, 512, 2048], reduce=False, s=2)
    X = bottelneck_residual_block(X, 3, [512, 512, 2048], reduce=False, s=2)

    # AVGPool
    X = AveragePooling2D((1, 1))(X)

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax')(X)

    model = Model(inputs=X_input, outputs=X, name='Resnet50')

    return model
