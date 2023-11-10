import sys, numpy as np
from keras.datasets import mnist
import os

(x_train, y_train), (x_test, y_test) = mnist.load_data()
images = x_train[0:1000].reshape(1000, 28 * 28) / 255
labels = y_train[0:1000]

one_hot_labels = np.zeros((len(labels), 10))

for index, label in enumerate(labels):
    one_hot_labels[index][label] = 1
labels = one_hot_labels

test_images = x_test.reshape(len(x_test), 28 * 28) / 255
test_labels = np.zeros((len(y_test), 10))
for index, label in enumerate(y_test):
    test_labels[index][label] = 1

np.random.seed(1)
relu = lambda x: (x >= 0) * x
relu2deriv = lambda x: x >= 0

alpha = 0.001
iters = 300
hidden_size = 100
pixels_per_image = 784
num_labels = 10
#Зададим размер для пакетного градиентного спуска
batch_size = 100

weights_0_1 = 0.2 * np.random.random((pixels_per_image, hidden_size)) - 0.1
weights_1_2 = 0.2 * np.random.random((hidden_size, num_labels)) - 0.1

prev_cnt = 0
j = 0
end_ahad = False
while ((j < iters) and end_ahad is False):
    error = 0.0
    correct_cnt = 0
    for i in range(int(len(images)/batch_size)):
        batch_start = i * batch_size
        batch_end = (i + 1) * batch_size
        # Берем картинку из images для обучения
        layer_0 = images[batch_start:batch_end]
        layer_1 = relu(np.dot(layer_0, weights_0_1))

        #Применяем маску прореживания
        dropout_mask = np.random.randint(2, size=layer_1.shape)
        layer_1 *= dropout_mask * 2

        layer_2 = np.dot(layer_1, weights_1_2)

        # Вычисление ошибки
        error += np.sum((labels[i:i + 1] - layer_2) ** 2)

        # Обучение, используя пакетный градиентный спуск
        for k in range(batch_size):
            pt1 = np.argmax(layer_2[k:k+1])
            pt2 = np.argmax(labels[batch_start+k:batch_start+k+1])
            correct_cnt += int(pt1 == pt2)

            layer_2_delta = (labels[batch_start:batch_end] - layer_2) / batch_size
            layer_1_delta = np.dot(layer_2_delta, weights_1_2.T) * relu2deriv(layer_1)
            layer_1_delta *= dropout_mask # <-  в обратном ходе
            weights_1_2 += alpha * np.dot(layer_1.T, layer_2_delta)
            weights_0_1 += alpha * np.dot(layer_0.T, layer_1_delta)
        sys.stdout.write("\r I:" + str(j) + " Train-Err:" + str(error / float(len(images)))[0:5] + " Train-Acc:" + str(
            correct_cnt / float(len(images))))

    if j % 10 == 0 or j == iters - 1:
        error = 0.0
        correct_cnt = 0  # Сравнение с тестовыми изображениями

        for i in range(len(test_images)):
            layer_0 = test_images[i:i + 1]
            layer_1 = relu(np.dot(layer_0, weights_0_1))
            layer_2 = np.dot(layer_1, weights_1_2)

            error += np.sum((test_labels[i:i + 1] - layer_2) ** 2)
            correct_cnt += int(np.argmax(layer_2) == np.argmax(test_labels[i:i + 1]))

        sys.stdout.write(" Test-Err:" + str(error / float(len(test_images)))[0:5] + " Test-Acc:" + str(
            correct_cnt / float(len(test_images))))

        print()

        # if prev_cnt > correct_cnt:
        #     end_ahad = True

        # prev_cnt = correct_cnt

    j += 1