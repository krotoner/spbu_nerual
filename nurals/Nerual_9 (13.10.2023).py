print("|\------------start--program--Neural_8------------\n")
import numpy as np


def main():
    logs("Calling function main", "new-log")

    alpha = 0.1

    weights = np.array([
        0.5,
        0.48,
        -0.5
    ])
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

    a = 0
    for i in range(80):
        general_error = 0
        for row_index in range(len(street_lights)):
            input_data = street_lights[row_index]
            goal_pred = walk_vs_stop[row_index]

            pred = np.dot(weights, input_data)
            error = (pred - goal_pred) ** 2
            general_error += error

            weight_data = (pred - goal_pred) * input_data * alpha
            weights -= weight_data
            print(f'Прогноз: {pred}\n Ошибка: {error}\n')
            a += 1

        print(f'{i}\nОбщая ошибка {general_error}\n')
    # print logs

    print('\n', "-" * 50, '\n')
    logs("", "print-log")
    logs("", "clear-logs")


def logs(massage, code):
    file = open('logs.txt', 'r')
    text = ""

    if code == "print-log":
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
