import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from tqdm.auto import tqdm
from catboost import CatBoostClassifier
from collections import Counter
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

tab = pd.read_csv(r"C:\Users\n.shamankov\Downloads\train.csv")
test = pd.read_csv(r"C:\Users\n.shamankov\Downloads\test.csv")


def train_simple_ensemble(train_data, test_data, estimator, num_folds=5, threshold=0.95, ret_pl=False):
    train_x, train_y = train_data
    test_x = test_data
    preds = []
    for train_idxs, val_idxs in StratifiedKFold(n_splits=num_folds, shuffle=True).split(train_x, train_y):
        train_x_, val_x_ = train_x[train_idxs], train_x[val_idxs]
        train_y_, val_y_ = train_y[train_idxs], train_y[val_idxs]
        model = estimator
        model.fit(train_x_, train_y_)
        preds.append(model.predict_proba(test_x))

        _preds = model.predict(val_x_)
        print(f'sklearn f1 score is {f1_score(val_y_, _preds)}')
    preds = np.stack(preds)
    #     K x N x 2
    null_preds = preds[:, :, 1].min(axis=0)
    one_preds = preds[:, :, 1].max(axis=0)
    mean_preds = preds[:, :, 1].mean(axis=0)
    if not ret_pl:
        return (mean_preds > 0.5).astype('int32')
    one_idxs, null_preds = np.where(one_preds > threshold)[0], np.where(null_preds < 1 - threshold)[0]
    new_idxs = np.concatenate([one_idxs, null_preds])
    test_x_ = test_x[new_idxs]
    test_y_ = np.zeros(len(test_x_))
    test_y_[:len(one_idxs)] = 1
    return (np.concatenate([train_x, test_x_]), np.concatenate([train_y, test_y_])), test_x[~new_idxs]


def get_xy(table, test_=False):
    data = table.values
    if test_:
        return data
    return data[:, :-1], data[:, -1]


def compare_with_best_score(best_path=r"C:\Users\n.shamankov\submission.csv"):
    best_tab = pd.read_csv(best_path)
    best_res, res = best_tab.to_numpy(), test_dt.to_numpy()
    best_res, res = best_res[:, 0], res[:, 0]
    num_eq = [True if best_res[i] == res[i] else False for i in range(len(res))]
    num_tp = Counter(num_eq)
    return num_tp[True] / (num_tp[True] + num_tp[False])


X_train, Y_train = get_xy(tab)
X_test = get_xy(test, test_=True)

# model1 =  KNeighborsClassifier(n_neighbors = 3,
#                               leaf_size = 10
#                               #weights = 'distance'
#                              )
# model2 = svm.NuSVC(nu = 0.15, tol= 1e-4, probability=True, degree = 3)
# model3 = MLPClassifier(alpha = 0.005,
#                       learning_rate = 'adaptive',
#                       early_stopping = True
#                      )
# model3 = svm.SVC(tol= 5e-4, probability=True, random_state=10, degree = 2)
# model = VotingClassifier(estimators=[('lr', model1), ('rf', model2), ('gnb', model3)], voting='soft')
model = QuadraticDiscriminantAnalysis(reg_param=0.45)

num_pseudo_label_steps = 7

X_train_, Y_train_, X_test_ = X_train, Y_train, X_test
print('Start test size:', len(X_test))
for step in range(num_pseudo_label_steps):
    (X_train_, Y_train_), X_test_ = train_simple_ensemble((X_train_, Y_train_), X_test_, model, 5, 0.99, ret_pl=True)
    print(f'Step f{step} test size:', len(X_test_))

preds = train_simple_ensemble((X_train_, Y_train_), X_test, model, 5, 0.99, ret_pl=False)

test_dt = pd.DataFrame()
test_dt['target'] = preds
test_dt.to_csv('submission.csv', index=False)
print(compare_with_best_score())
