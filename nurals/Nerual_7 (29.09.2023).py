import numpy


def network_nural(input_date, weigh):
    pred = input_date * weigh
    return pred


weigh = 0.5
input_date = 0.5
goal_pred = 0.8
step_amount = 10

for iterration in range(2101):
    pred = input_date * weigh
    error = (pred - goal_pred) ** 2

    print(f'С предсказанием: {pred} ошибка: {error}')
    up_predition = input_date * (weigh + step_amount)
    up_error = (goal_pred - up_predition) ** 2

    down_predition = input_date * (weigh - step_amount)
    down_error = (goal_pred - up_predition) ** 2

    if up_error > down_error:
        weigh -= step_amount
    else:
        weigh += step_amount
