import utils
import pandas as pd
import numpy as np

def read_csv(filename):
    f = open(filename, 'r')
    lines = f.readlines()
    matrix = []
    for line in lines:
        l = line.split(',')
        arr = []
        for w in l:
            arr.append(float(w))
        matrix.append(arr)
    return matrix


if __name__ == '__main__':

    ### BOSTON REGRESSION
    # X_added = np.array(read_csv('W_est_boston.csv'))
    # X = np.array(read_csv('W_est_boston_original.csv'))

    ### METABRIC REGRESSION
    X_added = np.array(read_csv('W_est_metabric.csv'))
    X = np.array(read_csv('W_est_metabric_original.csv'))

    # ### METABRIC CLASSIFICATION
    # X_added = np.array(read_csv('W_est_metabric_classification.csv'))
    # X = np.array(read_csv('W_est_metabric_original.csv'))

    print(X_added.shape)
    print(X.shape)
    
    acc = utils.count_accuracy(X, X_added != 0)

    print(acc)