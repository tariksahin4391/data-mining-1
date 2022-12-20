import math
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pydotplus
from sklearn import tree
from IPython.display import Image


# array içindeki elemanları gruplar
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


# entropy hesaplayan fonksiyon
def calculate_entropy(decision_array):
    grouped = group_array(decision_array)
    total_elem_count = len(decision_array)
    total_entropy = 0
    for elem in grouped:
        total_entropy = total_entropy + (elem[1] / total_elem_count) * math.log2(elem[1] / total_elem_count)
    total_entropy = -1 * total_entropy
    return total_entropy


def get_information_gain_for_numerical_features(samples, decisions, from_decision_tree=True):
    general_entropy = calculate_entropy(decisions)
    samples_and_indexes = []
    for j in range(0, len(samples)):
        samples_and_indexes.append([samples[j], decisions[j]])
    # numeric değerler değerlendirilmeden önce küçükten büyüğe sıralanır
    samples_and_indexes.sort(key=lambda x: x[0])
    sample_arr = list(map(lambda x: x[0], samples_and_indexes))
    class_arr = list(map(lambda x: x[1], samples_and_indexes))
    entropy_array, val = calculate_entropy_for_numerical_array(sample_arr, class_arr, from_decision_tree)
    if not from_decision_tree:
        print('total entropy ', entropy_array[0][0])
        print('total information gain ', general_entropy - entropy_array[0][0])
    return general_entropy - entropy_array[0][0], val


# class değerleri numerik dataya göre sıralanmıştı. şimdi bu class değerlerini baştan sona gezerek en optimum noktadan
# bölüyoruz
def calculate_entropy_for_numerical_array(sample_arr, result_arr, from_decision_tree):
    entropy_and_index_array = []
    for i in range(0, len(result_arr)):
        if i < len(result_arr) - 1 and result_arr[i] != result_arr[i + 1] and sample_arr[i] != sample_arr[i + 1]:
            index = i
            part1 = result_arr[0: i + 1]
            part2 = result_arr[i + 1: len(result_arr)]
            entropy = calculate_numerical_arr_entropy(part1, len(result_arr)) + calculate_numerical_arr_entropy(part2,
                                                                                                                len(result_arr))
            entropy_and_index_array.append([entropy, index])
            # print('if we choose index ', index, ' and value ', sample_arr[index], ' entropy will be ', entropy)
    if len(entropy_and_index_array) == 0:
        return [[0, len(result_arr) - 1]], sample_arr[len(sample_arr) - 1]
    entropy_and_index_array.sort(key=lambda x: x[0])
    if not from_decision_tree:
        print('we should split at index ', entropy_and_index_array[0][1], ' and value ', sample_arr[entropy_and_index_array[0][1]])
    return entropy_and_index_array, sample_arr[entropy_and_index_array[0][1]]


# bölünmüş noktanın(split) entrpy değerini hesaplar
def calculate_numerical_arr_entropy(arr, total_arr_count):
    grouped = group_array(arr)
    entropy = 0
    for elem in grouped:
        entropy = entropy + ((elem[1] / len(arr)) * (math.log2(elem[1] / len(arr))))
    entropy = (entropy * -1) * (len(arr) / total_arr_count)
    return entropy


def get_maximum_information_gain_for_categorical_features(sample_arr, decisions, from_generate_tree=True):
    general_entropy = calculate_entropy(decisions)
    total_elem_count = len(decisions)
    # kararların gruplanması
    grouped_samples = group_array(sample_arr)  # sample arrayin kendi içinde gruplanması
    if not from_generate_tree:
        print('grouped elems ', grouped_samples)
    # I(1,3,2,...)
    local_gain = []
    grouped_decisions = group_array(decisions)
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
    if not from_generate_tree:
        print('sample distribution I(X,Y,....)')
        print(local_gain)
        print("total entropy ", total)
    gain = general_entropy - total
    if not from_generate_tree:
        print("total information gain ", gain)
    return gain


df = pd.read_csv('teleCust1000t.csv')
columns = df.columns.values

feature_columns = columns[0:len(columns) - 1]
class_column = columns[len(columns) - 1]
class_array = df[class_column].values
categorical_features = [0, 3, 6, 8, 9, 10]
print('----categorical features----')
for i in categorical_features:
    print(feature_columns[i])
numerical_features = [1, 2, 4, 5, 7]
print('-----numerical features----')
for i in numerical_features:
    print(feature_columns[i])
class_entropy = calculate_entropy(class_array)
print('class entropy D(Info) : ', class_entropy)

analyzed_features = []
print('\n-----analyzing categorical features------')
categorical_and_numerical_feature_and_gain_array = []
for i in categorical_features:
    print('\n-------analyzing feature : ', feature_columns[i], '---------')
    gain = get_maximum_information_gain_for_categorical_features(df[feature_columns[i]].values, class_array, from_generate_tree=False)
    categorical_and_numerical_feature_and_gain_array.append([feature_columns[i], gain, -1])
    analyzed_features.append(feature_columns[i])
print('\n------analyzing numerical features------')
for i in numerical_features:
    print('\n-------analyzing feature : ', feature_columns[i], '---------')
    gain, val = get_information_gain_for_numerical_features(df[feature_columns[i]].values, class_array, from_decision_tree=False)
    categorical_and_numerical_feature_and_gain_array.append([feature_columns[i], gain, val])
    analyzed_features.append(feature_columns[i])
categorical_and_numerical_feature_and_gain_array.sort(key=lambda x: x[1], reverse=True)
print('\n-----information gains for each feature-------')
for i in categorical_and_numerical_feature_and_gain_array:
    print(i[0], ' : ', i[1])


#                            hangi elemanlar
def generate_tree(feature_name, indexes, step, numerical, split):
    str = ''
    for i in range(0,step):
        str = str + '-'
    str = str + feature_name
    print(str)
    if step == 4:
        # reached leaf node
        dummy = []
    else:
        sample_arr = df[feature_name].values
        if numerical:
            left_indexes = []
            right_indexes = []
            left_classes = []
            right_classes = []
            for i in indexes:
                if sample_arr[i] <= split:
                    left_indexes.append(i)
                    left_classes.append(class_array[i])
                else:
                    right_indexes.append(i)
                    right_classes.append(class_array[i])
            # sol ve sağ için ayrı ayrı gain hesapla ve fonksiyonu recursive çağır
            left_max_gain = []
            right_max_gain = []
            # left
            for i in categorical_features:
                if feature_columns[i] == feature_name:
                    continue
                filtered_samples = []
                for j in left_indexes:
                    filtered_samples.append(sample_arr[j])
                gain = get_maximum_information_gain_for_categorical_features(filtered_samples, left_classes)
                left_max_gain.append([feature_columns[i], gain, -1])
                analyzed_features.append(feature_columns[i])
            for i in numerical_features:
                if feature_columns[i] == feature_name:
                    continue
                filtered_samples = []
                for j in left_indexes:
                    filtered_samples.append(sample_arr[j])
                gain, val = get_information_gain_for_numerical_features(filtered_samples, left_classes)
                left_max_gain.append([feature_columns[i], gain, val])
                analyzed_features.append(feature_columns[i])
            left_max_gain.sort(key=lambda x: x[1], reverse=True)
            generate_tree(left_max_gain[0][0], left_indexes, step + 1, left_max_gain[0][2] != -1, left_max_gain[0][2])
            # right
            for i in categorical_features:
                if feature_columns[i] == feature_name:
                    continue
                local_samples = df[feature_columns[i]].values
                filtered_samples = []
                for j in right_indexes:
                    filtered_samples.append(local_samples[j])
                gain = get_maximum_information_gain_for_categorical_features(filtered_samples, right_classes)
                right_max_gain.append([feature_columns[i], gain, -1])
                analyzed_features.append(feature_columns[i])
            for i in numerical_features:
                if feature_columns[i] == feature_name:
                    continue
                filtered_samples = []
                local_samples = df[feature_columns[i]].values
                for j in right_indexes:
                    filtered_samples.append(local_samples[j])
                gain, val = get_information_gain_for_numerical_features(filtered_samples, right_classes)
                right_max_gain.append([feature_columns[i], gain, val])
                analyzed_features.append(feature_columns[i])
            right_max_gain.sort(key=lambda x: x[1], reverse=True)
            generate_tree(right_max_gain[0][0], right_indexes, step + 1, right_max_gain[0][2] != -1, right_max_gain[0][2])
        else:
            filtered_samples = []
            for i in indexes:
                filtered_samples.append(sample_arr[i])
            groups = list(map(lambda x: x[0], group_array(filtered_samples)))
            for g in groups:
                local_classes = []
                local_indexes = []
                for i in indexes:
                    if sample_arr[i] == g:
                        local_indexes.append(i)
                        local_classes.append(class_array[i])
                max_gain = []
                for i in categorical_features:
                    if feature_columns[i] == feature_name:
                        continue
                    filtered_samples = []
                    local_samples = df[feature_columns[i]].values
                    for j in local_indexes:
                        filtered_samples.append(local_samples[j])
                    gain = get_maximum_information_gain_for_categorical_features(filtered_samples, local_classes)
                    max_gain.append([feature_columns[i], gain, -1])
                    analyzed_features.append(feature_columns[i])
                for i in numerical_features:
                    if feature_columns[i] == feature_name:
                        continue
                    filtered_samples = []
                    local_samples = df[feature_columns[i]].values
                    for j in local_indexes:
                        filtered_samples.append(local_samples[j])
                    gain, val = get_information_gain_for_numerical_features(filtered_samples, local_classes)
                    max_gain.append([feature_columns[i], gain, val])
                    analyzed_features.append(feature_columns[i])
                max_gain.sort(key=lambda x: x[1], reverse=True)
                generate_tree(max_gain[0][0], local_indexes, step + 1, max_gain[0][2] != -1, max_gain[0][2])


init_indexes = []
for i in range(0, 800):
    init_indexes.append(i)
generate_tree(categorical_and_numerical_feature_and_gain_array[0][0], init_indexes, 1,
              categorical_and_numerical_feature_and_gain_array[0][2] != -1,
              categorical_and_numerical_feature_and_gain_array[0][0])


X = df[feature_columns].values
y = df[class_column].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=4)
print('\nTrain set:', X_train.shape, y_train.shape)
print('Test set:', X_test.shape, y_test.shape)
print('\n')

decisiontree_entropy = DecisionTreeClassifier(criterion='entropy', random_state=0, max_depth=4, max_leaf_nodes=16)

model_entropy = decisiontree_entropy.fit(X_train, y_train)

dot_data = tree.export_graphviz(model_entropy, out_file="resume.dot",
                                feature_names=feature_columns, class_names=['1', '2', '3', '4'],
                                filled=True, rounded=True, special_characters=True, leaves_parallel=False)

graph = pydotplus.graphviz.graph_from_dot_file("resume.dot")
graph.write_png('tree.png')

Image(filename='tree.png')
