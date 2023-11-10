print("|\------------start--program--Neural_8------------\n")
import numpy as np


def main():
    logs("Calling function main", "new-log")

    weight = np.array([
        [-0.5, 0.1, 0.7],  # pred1
        [0.21, -0.45, 0.6],  # pred2
        [0.39, 0.1, -0.25]  # pred3
    ])
    alpha = 1  # 0.1

    # входные данные игр
    games_num = np.array([0.3, 0.4, 0.6])
    fans = np.array([0.74, 0.22, 0.66])
    win_to_losses = np.array([0.8, 0.22, 0.44])

    # итоги игр
    wins = np.array([0.9, 0.76, 0.86])
    trauma = np.array([0.22, 0.44, 0.88])
    sedness = np.array([0.1, 0.05, 0.03])

    for i in range(3):
        input_date = np.array([games_num[i], fans[i], win_to_losses[i]])
        goal_pred = np.array([wins[i], trauma[i], sedness[i]])
        pred = np.dot(weight, input_date)
        error = (pred - goal_pred) ** 2

        # Градиентный спуск
        direction_and_mount = (pred - goal_pred) * input_date[i] * alpha
        # direction_and_mount[1] = 0

        weight -= direction_and_mount
        print(f'Ошибка: {error} Прогноз: {pred} ')
    print(weight)

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
print("--------------end-program-------------------------/|")
