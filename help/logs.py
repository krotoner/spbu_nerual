import sys, numpy as np
from keras.datasets import mnist

#Загрузка обучающих и тестовых заданий
(x_train, y_train), (x_test, y_test) = mnist.load_data()

images = (x_train[0:1000].reshape(1000, 28 * 28) / 255)
labels = (y_train[1:1000])

one_hot_labels = np.zeros((len(labels), 10))

for index, elem in enumerate(labels):
    one_hot_labels[index][elem] = 1
labels = one_hot_labels
test_images = x_test[0:1000].reshape(1000, 28 * 28) / 255
test_labels = np.zeros((len(y_test), 10))
for index, elem in enumerate(y_test):
    test_labels[index][elem] = 1

np.random.seed(1)
relu = lambda x: (x >= 0) * x
relu2driv = lambda x: x >= 0

alpha = 0.005
iters = 350
hidden_size = 40
pixel_per_image = 784
num_labels = 10

weights_0_1 = 0.2 * np.random.random((pixel_per_image, hidden_size))  - 0.1
weights_1_2 = 0.2 * np.random.random((hidden_size, num_labels)) - 0.1

for j in range(iters):
    error = 0.0
    correct_cnt = 0

    for i in range(len(images)):
        layer_0 = images[i:i+1]
        layer_1 = relu(np.dot(layer_0, weights_0_1))
        layer_2 = np.dot(layer_1, weights_1_2)

        error += np.sum((labels[i:i+1] - layer_2) ** 2)
        correct_cnt += int(np.argmax(layer_2) == np.argmax(labels[i:i+1]))

        layer_2_delta = labels[i:i+1] - layer_2
       # layer_1_delta =  np.dot(layer_2_delta, weights_1_2) + relu2driv(layer_1)
        layer_1_delta = np.dot(layer_2_delta, weights_1_2) + relu2driv(layer_1)
        weights_0_1 +=alpha * np.dot(layer_2_delta, layer_1)
        weights_1_2 += alpha * np.dot(layer_1_delta, layer_0)

    error = str(error/float(len(images)))[:5]
    correct_cnt = correct_cnt/float(len(images))
    sys.stdout.write(f'\nI: {j} E: {error} C: {correct_cnt}')