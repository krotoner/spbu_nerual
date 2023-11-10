import numpy as np

print("|\------------start--program--Neural_6------------\n")

arr_1 = np.array([22, 3, 4.5], dtype=float)
print(type(arr_1[2]))
print(arr_1[2])
print('\n', "--" * 25, '\n')

a = np.array([0, 1, 2, 3])
b = np.array([4, 5, 6, 7])
c = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
d = np.zeros(((3, 3)))
e = np.random.rand(0, 0)
print('a= ', a)
print('b= ', b)
print('c= ', c)
print('d= ', d)
print('e= ', e)
print('\n', "--" * 25, '\n')

a1 = np.zeros(((4, 1))).T
a2 = np.zeros(((4, 3)))
a3 = np.dot(a1, a2)  # ПРИЧЕМ ТУТ ПАРАМЕТР OUT=

print(f'Размер матрицы: {a3.shape} ')
print('\n', "--" * 25, '\n')


def main():
    logs("Calling function main", "new-log")

    # data

    weights_i_hid = [
        [0.1, 0.2, -0.1],
        [-0.1, 0.1, 0.9],
        [0.1, 0.4, 0.1]
    ]

    weights_hid_pred = [
        [0.1, 0.2, -0.1],
        [-0.1, 0.1, 0.9],
        [0.1, 0.4, 0.1]
    ]

    wins_num = [0.65, 0.8, 0.87]
    toes = [8.5, 9.5, 9.9]
    wl_rec = [0.65, 0.8, 0.8]
    fans = [1.2, 1.3, 0.5]

    # start program

    for i in range(len(wins_num)):
        input_data = [toes[i], wl_rec[i], fans[i]]
        pred = nerual_network(input_data, weights_i_hid, weights_hid_pred)
        print(pred)

    # print logs

    print('\n', "--" * 25, '\n')
    logs("", "print-log")
    logs("", "clear-logs")


def nerual_network(input_data, weights_i_hid, weights_hid_pred):
    logs("Calling function nerual_network", "new-log")
    hidden = np.dot(weights_hid_pred, input_data)
    prediction = np.dot(weights_i_hid, hidden)
    return prediction


def logs(massage, code):
    file = open('../logs.txt', 'r')
    text = ""

    if code == "print-log":
        print("logs:")
        print(file.read())
    else:
        text = file.read()
        text += massage
        text += '\n'
    file.close()

    file = open('../logs.txt', 'w')
    if code == "new-log":
        file.write(text)
    if code == "clear-log":
        file.write("")
    file.close()


if __name__ == '__main__':
    main()
print("\n--------------end-program-------------------------/|")
