print("|\------------start--program--Neural_8------------\n")
import numpy as np


def main():
    logs("Calling function main", "new-log")

    np.random.seed(1)
    alpha = 0.2
    hidden_size = 4

    street_lights = np.array([
        [1, 0, 1],
        [0, 1, 1],
        [0, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
        [1, 0, 1]
    ])

    walk_vs_stop = np.array([
        [0],
        [1],
        [0],
        [1],
        [1],
        [0]
    ])

    weights_0_1 = 2 * np.random.random((3, hidden_size)) - 1
    weights_1_2 = 2 * np.random.random((hidden_size, 1)) - 1

    for iter in range(60):
        layer_2_error = 0
        for i in range(len(street_lights)):
            layer_0 = street_lights[i:i + 1]
            layer_1 = relu(np.dot(layer_0, weights_0_1))
            layer_2 = np.dot(layer_1, weights_1_2)

            layer_2_error += np.sum((layer_2 - walk_vs_stop[i:i + 1] ** 2))

            layer_2_delta = (walk_vs_stop[i:i + 1] - layer_2)
            layer_1_delta = np.dot(layer_2_delta, weights_1_2.T) * relu2deriv(layer_1)

            weights_1_2 += alpha * np.dot(layer_1.T, layer_2_delta)
            weights_0_1 += alpha * np.dot(layer_0.T, layer_1_delta)

        if iter % 10 == 9:
            print(f'error: {layer_2_error}')

    # print logs
    #logs("", "print-log")
    logs("", "clear-logs")


def relu(x):
    logs("Calling function relu", "new-log")
    return (x > 0) * x


def relu2deriv(output):
    logs("Calling function relu2deriv", "new-log")
    return output > 0


def logs(massage, code):
    file = open('logs.txt', 'r')
    text = ""

    if code == "print-log":
        print('\n', "-" * 50)
        print("logs:")
        print(file.read())
    else:
        text = file.read()
        text += massage
        text += '\n'
    file.close()

    file = open('logs.txt', 'w')
    if code == "new-log":
        file.write(text)
    if code == "clear-log":
        file.write("")
    file.close()


if __name__ == '__main__':
    main()
print("\n--------------end-program-------------------------/|")
