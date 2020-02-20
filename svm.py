#!/usr/bin/env python3.7

from encode import create_vectors
from encode import train as encode
from sklearn.svm import OneClassSVM
from sklearn.svm import SVC
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
from reader import get_traces
import numpy as np
import os
import re
from random import shuffle


def one_class(log_name, train, test, nu, labels):
    ocsvm = OneClassSVM(kernel='linear', nu=nu)
    ocsvm.fit(train)
    classes = list(ocsvm.predict(test))

    f1 = f1_score(labels, classes)
    try:
        auc = roc_auc_score(labels, classes)
    except ValueError:
        auc = "n/a"
    acc = accuracy_score(labels, classes)
    prc = precision_score(labels, classes)
    rec = recall_score(labels, classes)
    # acc = (anom_checked.count(False) + normal_checked.count(True)) / float(len(test)) 

    print(log_name, 
        "OCSVM", 
        nu, 
        f1, auc, acc, prc, rec, sep=",")


def supervised(log_name, train, test, train_labels, test_labels):
    svc = SVC(kernel='linear')
    try:
        svc.fit(train, train_labels)
    except ValueError:
        return
    
    classes = list(svc.predict(test))

    f1 = f1_score(test_labels, classes)
    try:
        auc = roc_auc_score(test_labels, classes)
    except ValueError:
        auc = "n/a"
    acc = accuracy_score(test_labels, classes)
    prc = precision_score(test_labels, classes)
    rec = recall_score(test_labels, classes)
    # acc = (anom_checked.count(False) + normal_checked.count(True)) / float(len(test))

    print(log_name, 
          "SVM", 
          "",
          f1, auc, acc, prc, rec, sep=",")


def lof(log_name, train, test, k, contamination, labels):
    model = LocalOutlierFactor(k, contamination=contamination, novelty=True)
    model.fit(train)
    classes = list(model.predict(test))

    f1 = f1_score(labels, classes)
    try:
        auc = roc_auc_score(labels, classes)
    except ValueError:
        auc = "n/a"
    acc = accuracy_score(labels, classes)
    prc = precision_score(labels, classes)
    rec = recall_score(labels, classes)

    print(log_name, 
        "lof", 
        f'{k} - {contamination}', 
        f1, auc, acc, prc, rec, sep=",")


def train(log_name, traces_name, w2v_model):
    # normal_traces = [line.strip('\n') for line in open(f"traces/{traces_name}.txt", 'r')]
    traces, labels = get_traces(log_name)
    shuffle(traces)
    v = create_vectors(w2v_model, traces)


    limit = int(len(traces) * 0.7)
    train, test = v[:limit], v[limit:]
    train_labels, test_labels = labels[:limit], labels[limit:]
    # train_traces, test_traces = traces[:limit], traces[limit:]
    
    # print("log_name,nu,method,accuracy")

    # train_labels = np.array([1 if is_normal(trace, normal_traces) else -1 for trace in traces[:limit]])
    # test_labels = np.array([1 if is_normal(trace, normal_traces) else -1 for trace in traces[limit:]])

    for nu in [0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5]: 
        one_class(log_name, train, test, nu, test_labels)
    
    for k in [10, 15, 20, 25]:
        for contamination in [0.01, 0.1, 0.3]:
            lof(log_name, train, test, k, contamination, test_labels)

    supervised(log_name, train, test, train_labels, test_labels)


def is_normal(vector, normal_traces):
    # if true, it is a normal behaviour
    return " ".join(vector) in normal_traces


if __name__ == "__main__":
    # anom_types = ["all", "rework", "earlylate", "skip", "insert"]
    # proportions = [5, 10, 15, 20, 30]
    
    # for i in [1,2]: 
    #     for a in anom_types:
    #         for p in proportions:
    #             log_name = f"log{i}_anom_{a}_{p}"
    #             w2v_model = encode(log_name, min_count=1)
    #             train(log_name, f"normal_pn{i}", w2v_model)
    for f in os.listdir('logs/csv'):
        w2v_model = encode('logs/csv/' + f, min_count=1)
        train('logs/csv/' + f, None, w2v_model)
