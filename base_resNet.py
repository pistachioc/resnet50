import numpy as np
from keras.layers import Input, Dense, Add, Activation, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D
from keras.models import Model, load_model


def bottelneck_residual_block(X, kernel_sizes, filters=[], reduce=False, s=2):
    """Ham khoi tao khoi reisdual, chon reduce hay khong voi 3 khoi conv o mainpath
    X: tensor dau vao
    kernel_sizes: kich thuoc kernel cua lop conv o giua cua mainpath
    filters: mang so nguyen chua so filters cua tung lop conv
    reduce:  True-co them conv o shortcutpath hay khong"""

    X_shortcut = X
    F1, F2, F3 = filters

    if reduce:
        X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s))(X_shortcut)
        X_shortcut = BatchNormalization(axis=3)(X_shortcut)

        # ta cau hinh cho strides cua layer dau tien giong voi shortcut path
        X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid')(X)
        X = BatchNormalization(axis=3)(X)
        X = Activation('relu')(X)

    else:
        X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid')(X)
        X = BatchNormalization(axis=3)(X)
        X = Activation('relu')(X)

    # thanh phan thu 2 cua mainpath
    X = Conv2D(filters=F2, kernel_size=kernel_sizes, strides=(1, 1), padding='same')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    # thanh phan thu 3 cua mainpath
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid')(X)
    X = BatchNormalization(axis=3)(X)

    # Cong 2 path
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X