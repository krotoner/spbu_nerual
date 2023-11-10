print("|\----------------start--programm----------------------||\n")

# сеть с 3 входами и одним выходом

def nerual_network (input, weight):
    # сама сеть
    prediction = w_sum(input,weight)
    return prediction

def w_sum (input, weight):
    sum = 0
    # первый вариант
    # for i in range(len(input)):
    #     sum = input[i] * weight[i]

    # второй вариант
    for i, j in zip(input, weight):
        sum += i * j

    return sum


weight = [0.1, 0.2, 0.3, 0]
toes = [8.5, 9.5, 9.9]
wlrec = [0.65, 0.8, 0.8]
fans = [1.2, 1.3, 0.5]

for i in range(3):
    input_date = [weight[i], toes[i], wlrec[i], fans[i]]
    pred = nerual_network(input_date, weight)
    print(f'при значениях {input_date} предсказания будут {pred}')

print("\n||----------end-programm----------------------------/|")