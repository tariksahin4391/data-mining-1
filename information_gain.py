import math

import util

""""
def get_information_gains_for_numerical_features(samples, decisions):
    samples_and_indexes = []
    for i in range(0, len(samples)):
        samples_and_indexes.append([])
        for j in range(0, len(samples[i])):
            samples_and_indexes[i].append([samples[i][j], decisions[j]])
    for elem in samples_and_indexes:
        elem.sort(key=lambda x: x[0])
    for s in samples_and_indexes:
        sample_arr = list(map(lambda x: x[0], s))
        class_arr = list(map(lambda x: x[1], s))
        print(sample_arr)
        print(class_arr)
        gain = calculate_entropy_for_numerical_array(sample_arr,class_arr)
        print("-------gain-------")
        print(gain)
    print("ok")
"""


def get_information_gain_for_numerical_features(samples, decisions):
    samples_and_indexes = []
    for j in range(0, len(samples)):
        samples_and_indexes.append([samples[j], decisions[j]])
    samples_and_indexes.sort(key=lambda x: x[0])
    sample_arr = list(map(lambda x: x[0], samples_and_indexes))
    class_arr = list(map(lambda x: x[1], samples_and_indexes))
    print(sample_arr)
    print(class_arr)
    gain = calculate_entropy_for_numerical_array(sample_arr, class_arr)
    print("-------gain-------")
    print(gain)
    return gain


# class değerleri numerik dataya göre sıralanmıştı. şimdi bu class değerlerini baştan sona gezerek en optimum noktadan
# bölüyoruz
def calculate_entropy_for_numerical_array(sample_arr, result_arr):
    entropy_and_index_array = []
    for i in range(0, len(result_arr)):
        if i < len(result_arr) - 1 and result_arr[i] != result_arr[i + 1] and sample_arr[i] != sample_arr[i + 1]:
            index = i
            part1 = result_arr[0: i + 1]
            part2 = result_arr[i + 1: len(result_arr)]
            entropy = calculate_numerical_arr_entropy(part1, len(result_arr)) + calculate_numerical_arr_entropy(part2,
                                                                                                                len(result_arr))
            entropy_and_index_array.append([entropy, index])
    if len(entropy_and_index_array) == 0:
        return [[0, len(result_arr) - 1]]
    return entropy_and_index_array


def calculate_numerical_arr_entropy(arr, total_arr_count):
    grouped = util.group_array(arr)
    entropy = 0
    for elem in grouped:
        entropy = entropy + ((elem[1] / len(arr)) * (math.log2(elem[1] / len(arr))))
    entropy = (entropy * -1) * (len(arr) / total_arr_count)
    return entropy


def get_maximum_information_gain_for_categorical_features(sample_arr, decisions):
    general_entropy = calculate_entropy(decisions)
    total_elem_count = len(decisions)
    # kararların gruplanması
    grouped_samples = util.group_array(sample_arr)  # sample arrayin kendi içinde gruplanması
    # I(1,3,2,...)
    local_gain = []
    grouped_decisions = util.group_array(decisions)
    for grouped_sample in grouped_samples:
        group_gain = []
        for grouped_decision in grouped_decisions:
            count = 0
            for j in range(0, len(decisions)):
                if sample_arr[j] == grouped_sample[0] and decisions[j] == grouped_decision[0]:
                    count = count + 1
            group_gain.append(count)
        local_gain.append(group_gain)
    total = 0
    for elem in local_gain:  # [3,2]
        elem_gain_total = 0
        elem_total = sum(elem)
        for e in elem:
            if not e == 0:
                elem_gain_total = elem_gain_total + ((e / elem_total) * math.log2((e / elem_total)))
        elem_gain_total = elem_gain_total * -1
        elem_gain_total = (elem_total / total_elem_count) * elem_gain_total
        total = total + elem_gain_total
    print(local_gain)
    print("total entropy ", total)
    gain = general_entropy - total
    print("total information gain ", gain)
    return gain


def calculate_entropy(decision_array):
    grouped = util.group_array(decision_array)
    # total_elem_count = sum([g[1] for g in grouped])
    # print(total_elem_count)
    total_elem_count = len(decision_array)
    total_entropy = 0
    for elem in grouped:
        total_entropy = total_entropy + (elem[1] / total_elem_count) * math.log2(elem[1] / total_elem_count)
    total_entropy = -1 * total_entropy
    return total_entropy


def calculate_entropy_by_possibility(possibilities):
    total_elem_count = sum(possibilities)
    total_entropy = 0
    for i in possibilities:
        total_entropy = total_entropy + ((i / total_elem_count) * math.log2(i / total_elem_count))
    return -1 * total_entropy


samples = [5, 1, 2, 3, 2, 8, 4]
categorical = ['A', 'A', 'A', 'B', 'A', 'B', 'B']
decisions = [1, 2, 2, 2, 2, 3, 1]

get_information_gain_for_numerical_features(samples, decisions)
get_maximum_information_gain_for_categorical_features(categorical,decisions)
