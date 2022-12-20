import math


def calculate_mean(arr):
    x = 0
    for a in arr:
        x = x + a
    return x / len(arr)


def find_median(arr):
    if len(arr) % 2 == 0:
        half = math.floor((len(arr) / 2))  # orta elemanı bul
        exactHalf = half - 0.5
        res = (arr[half] + arr[half - 1]) / 2
    else:
        half = math.floor(len(arr) / 2)
        exactHalf = half
        res = arr[half]
    return res, half, exactHalf


def calculate_variance(arr):
    mean = calculate_mean(arr)
    total = 0
    for a in arr:
        total = total + pow((a - mean), 2)
    return total / (len(arr) - 1)


def calculate_covariance_value_between_arrays(arr1, arr2):
    mean1 = calculate_mean(arr1)
    mean2 = calculate_mean(arr2)
    total = 0
    for i in range(0, len(arr1)):
        total = total + ((arr1[i] - mean1) * (arr2[i] - mean2))
    return round(total / (len(arr1) - 1), 4)


def calculate_covariance_matrix(arr_list):
    result = []
    for i in range(0, len(arr_list)):
        result.append([])
        for j in range(0, len(arr_list)):
            if j < i:
                result[i].append(result[j][i])
            else:
                result[i].append(calculate_covariance_value_between_arrays(arr_list[i], arr_list[j]))
    return result


def calculate_absolut(x):
    if x >= 0:
        return x
    else:
        return x * (-1)


def group_array(arr):
    result_as_list = []
    for i in arr:
        if len(result_as_list) == 0:
            result_as_list.append([i, 1])
        else:
            found = False
            for e in result_as_list:
                if e[0] == i:
                    e[1] = e[1] + 1
                    found = True
                    break
            if not found:
                result_as_list.append([i, 1])
    result_as_tuple_list = []
    for r in result_as_list:
        result_as_tuple_list.append((r[0], r[1]))
    return result_as_tuple_list
