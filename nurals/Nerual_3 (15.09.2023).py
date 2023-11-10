print("|\------------start--program--Nerual_3------------\n")


def main():
    logs("Colling function main", "new-log")

    weights = [
        [0.1, 0.1, -0.3],
        [0.1, 0.2, 0],
        [0, 1.3, 0.1]
        ]
    wins_num = [0.65, 0.8, 0.87]
    toes = [8.5, 9.5, 9.9]
    wlrec = [0.65, 0.8, 0.8]
    fans = [1.2, 1.3, 0.5]

    for i in range(0,3):
        input_data = [toes[i],wlrec[i],fans[i]]
        #pred = nerual_network(input_data, weights)
        #print(pred)

    for i in range(len(wins_num)):
        pred = nerual_network(input_data[i], weights)
        print(pred)

    print('\n', "--" * 25, '\n')
    logs("", "print-log")
    logs("", "clear-logs")


def nerual_network (input_data, weight):
    logs("Colling function nerual_network", "new-log")
    prediction = ele_mul(input_data, weight)
    #prediction = vect_mat_mul(input_data, weight)
    return prediction


def vect_mat_mul(input_data,weight):
    logs("Colling function vect_mat_mul", "new-log")
    temp_list = []
    for w_row in weight:
        temp_list.append(w_sum(input_data, w_row))

    return temp_list


def ele_mul(input_data,weight):
    logs("Colling function ele_mul", "new-log")
    temp_list = []
    for w in range(len(weight)):
        temp_list.append(input_data*w)
    return temp_list


def w_sum(input, weight):
    logs("Colling function w_sum", "new-log")
    sum = 0
    for i in range(len(input)):
        sum = input[i] * weight[i]

    return sum


def logs(massage, code):

    file = open('../logs.txt', 'r')

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
