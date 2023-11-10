import numpy


print("|\------------start--program--Neural_4------------\n")


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
    hidden = vect_mat_mul(input_data, weights_hid_pred)
    prediction = vect_mat_mul(hidden, weights_i_hid)
    return prediction


def vect_mat_mul(input_data, weight):
    logs("Calling function vect_mat_mul", "new-log")
    temp_list = []
    for w_row in weight:
        temp_list.append(w_sum(input_data, w_row))

    return temp_list


def ele_mul(input_data, weight):
    logs("Calling function ele_mul", "new-log")
    temp_list = []
    for w in range(len(weight)):
        temp_list.append(input_data*w)
    return temp_list


def w_sum(input_date, weight):
    logs("Calling function w_sum", "new-log")
    summ = 0
    for i in range(len(input_date)):
        summ = input_date[i] * weight[i]

    return summ


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
