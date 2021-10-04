import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def percentage(part, whole):
    return 100 * float(part) / float(whole)


def remove_zero_rows(X, Y):

    new_X = pd.DataFrame(X)
    new_X['sum'] = X.sum(axis=1)
    new_X = new_X[new_X["sum"] > 0]
    Y = Y[X.sum(axis=1) > 0]
    new_X.drop("sum", axis=1)

    return new_X, Y


def get_data(option=1, pose_1=True):
    if option == 1:
        #fnamelbl = "/home/carlos/Desktop/Genesis/Tesis_Master/paper/paper/Carlos/new_data/tox_data_02_2021.csv"
        fnamelbl = "./data/tox_data_02_2021.csv"
        dataset = []
        with open(fnamelbl) as f:
            names = f.readline().strip()
            names = names.split(',')
            names = names[3:]
            names.insert(0, 'id_pose')
            for row in f:
                row = row.strip()
                r = row.split(',')
                pose = r[1] + '_' + r[2]
                r = r[3:]
                for i in range(len(r)):
                    if len(r[i]) == 0:
                        r[i] = 0
                r.insert(0, pose)
                dataset.append(r)

        dataset = pd.DataFrame(dataset, columns=names)
        if pose_1:
            # only pose 1
            X = dataset[dataset['id_pose'].str.endswith('_1')]
            labels = X[['id_pose', 'active']]
            X = X.drop(['id_pose', 'active'], axis=1)
            X = X.apply(pd.to_numeric)
        else:
            labels = dataset[['id_pose', 'active']]
            X = dataset.drop(['id_pose', 'active'], axis=1)
            X = X.apply(pd.to_numeric)

        Y = labels['active']
        Y = Y.astype(int)

        for c in X.columns:
            total = X[c].unique()
            if len(total) == 1:
                X = X.drop(c, axis=1)

        return X, Y
    elif option == 2:
        fnamelbl = "/home/carlos/Desktop/Genesis/Tesis_Master/paper/paper/Carlos/new_data/table_aminoacid.csv"
        dataset = []

        with open(fnamelbl) as f:
            names = f.readline().strip()
            names = names.split(',')
            for row in f:
                row = row.strip()
                r = row.split(',')
                dataset.append(r)

        dataset = pd.DataFrame(dataset, columns=names)
        if pose_1:
            X = dataset[dataset['poses'].str.endswith('_1')]
            labels = X[['poses', 'active']]

            X = X.drop(['poses', 'active'], axis=1)
            X = X.apply(pd.to_numeric)
        else:
            labels = dataset[['poses', 'active']]
            X = dataset.drop(['poses', 'active'], axis=1)
            X = X.apply(pd.to_numeric)

        Y = labels['active']
        Y = Y.astype(int)

        for c in X.columns:
            total = X[c].unique()
            if len(total) == 1:
                X = X.drop(c, axis=1)

        return X, Y


# main function
def train_model(model, batch_size, learning, iter, optimizer):
    # dummy variables: model, batch_size, learning, iter and optimizer (please print them in the terminal)
    # current prints must be change into messages returned by the API to the Front-End

    # tox dataset
    X, Y = get_data(option=1, pose_1=True)

    counter = Counter(Y)

    print("inicial Classes count  0: " +
          str(counter[0]) + " 1: " + str(counter[1]))

    row_count = X.shape[0]
    toremove = []
    for c in X.columns:
        m_count = X[X[c] == 0][c].count()
        if m_count > 0:
            pc = percentage(m_count, row_count)
            if pc >= 91:
                toremove.append(c)

    print("Columns to be remove: " + str(len(toremove)))

    X.drop(toremove, inplace=True, axis=1)

    print("Total columns: " + str(len(X.columns)))

    X, Y = remove_zero_rows(X, Y)

    counter = Counter(Y)
    print("Rows after remove zeros  0: " +
          str(counter[0]) + " 1: " + str(counter[1]))

    X_train, X_validation, Y_train, Y_validation = train_test_split(
        X, Y, test_size=0.5, random_state=2, stratify=Y)

    counter = Counter(Y_train)

    print("Spliting dataset:  0: " +
          str(counter[0]) + " 1: " + str(counter[1]))

    print("running: KNeighborsClassifier")

    model = KNeighborsClassifier(n_neighbors=1, algorithm='ball_tree')
    model.fit(X_train, Y_train)
    predictions = model.predict(X_validation)

    # Results from the training
    # This three lines must be returned by the API to the Front-End
    print("Accuracy score: " + str(accuracy_score(Y_validation, predictions)))
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))

    acc = accuracy_score(Y_validation, predictions)
    acc_score = 'Accuracy score: {}'.format(acc)
    conf_matrix = confusion_matrix(Y_validation, predictions).tolist()
    class_report = classification_report(Y_validation, predictions)

    return acc_score, conf_matrix, class_report




