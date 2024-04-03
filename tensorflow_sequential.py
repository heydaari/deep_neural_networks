import tensorflow as tf
from tensorflow import keras

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocess data : all pixels in range 0 to 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Flattenning Images
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

print(type(x_train), x_train.shape)
print(type(x_test), x_test.shape)
print(type(y_train), y_train.shape)
print(type(y_test), y_test.shape)

# Defining Sequential Model
model = keras.models.Sequential([
    keras.layers.Dense(units = 512, activation = 'relu', input_shape = [28 * 28]),
    keras.layers.Dense(units = 512, activation = 'relu'),
    keras.layers.Dense(units = 10, activation = 'softmax')
])

model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'sgd', metrics=['accuracy'])

model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 10 )