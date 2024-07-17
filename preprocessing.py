from keras.datasets import cifar10
from keras.utils import to_categorical


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# input image dimension
input_shape = x_train.shape[1:]

# normalize data
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

num_classes = 10
data_augumentation = True

# Convert class vectors to binary class matrices.
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

