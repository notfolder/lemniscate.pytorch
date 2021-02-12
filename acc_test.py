from sklearn.metrics.cluster import pair_confusion_matrix
from sklearn.metrics import accuracy_score
import numpy as np

def print_acc(label_true, label_pred):
    matrix = pair_confusion_matrix(label_true, label_pred)
    print(matrix)
    acc = (matrix[0][0] + matrix[1][1])/sum(sum(matrix))
    print(acc)

    label_true = np.array(label_true)
    label_pred = np.array(label_pred)
    print(acc_score(label_true, label_pred))

def acc_score(label_true, label_pred):
    dic = {}
    for i in np.unique(label_pred):
        dic[i] = np.argmax(np.bincount(label_true[label_pred == i]))
    v = np.array(list(dic.values()))
    sidx = np.searchsorted(list(dic), label_pred)
    acc = accuracy_score(label_true, v[sidx])
    return acc

print_acc([1,1,0,0],[0,0,1,1])
print_acc([1,1,0,0],[0,0,2,2])
print_acc([0,0,0,1,1,1,2,2,2],
          [0,0,1,1,1,1,2,2,0])
print_acc([0,0,0,1,1,1,2,2,2],
          [2,2,0,0,0,1,1,1,2])
