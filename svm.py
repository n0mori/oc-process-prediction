#!/usr/bin/env python3.7

from encode import create_vectors
from encode import train as encode
from sklearn.svm import OneClassSVM
from sklearn.svm import SVC
from reader import get_traces
import numpy as np
import os
import re
from random import shuffle


def one_class(log_name, test_vectors, train, test, normal_traces, nu):
    ocsvm = OneClassSVM(kernel='linear', nu=nu)
    ocsvm.fit(train)
    classes = list(ocsvm.predict(test))

    anom_cases = [test_vectors[k] for k, x in enumerate(classes) if x == -1]
    normal_cases = [test_vectors[k] for k, x in enumerate(classes) if x == 1]

    anom_checked = [is_normal(t, normal_traces) for t in anom_cases]
    normal_checked = [is_normal(t, normal_traces) for t in normal_cases]

    acc = (anom_checked.count(False) + normal_checked.count(True)) / float(len(test)) 

    print(log_name, 
        "oc", 
        nu, 
        acc, sep=",")


def supervised(log_name, test_vectors, train, test, normal_traces, labels):
    svc = SVC(kernel='linear')
    svc.fit(train, labels)
    classes = list(svc.predict(test))

    anom_cases = [test_vectors[k] for k, x in enumerate(classes) if x == -1]
    normal_cases = [test_vectors[k] for k, x in enumerate(classes) if x == 1]

    anom_checked = [is_normal(t, normal_traces) for t in anom_cases]
    normal_checked = [is_normal(t, normal_traces) for t in normal_cases]

    acc = (anom_checked.count(False) + normal_checked.count(True)) / float(len(test))

    print(log_name, 
          "sv", 
          "",
          acc, sep=",")


def train(log_name, traces_name, w2v_model):
    normal_traces = [line.strip('\n') for line in open(f"traces/{traces_name}.txt", 'r')]
    traces = get_traces("logs/" + log_name + ".csv")
    shuffle(traces)
    v = create_vectors(w2v_model, traces)


    limit = int(len(traces) * 0.7)
    train, test = v[:limit], v[limit:]
    train_traces, test_traces = traces[:limit], traces[limit:]
    
    # print("log_name,nu,method,accuracy")

    labels = np.array([1 if is_normal(trace, normal_traces) else -1 for trace in traces[:limit]])

    for nu in [0.05, 0.1, 0.3, 0.5]: 
        one_class(log_name, test_traces, train, test, normal_traces, nu)

    supervised(log_name, test_traces, train, test, normal_traces, labels)


def is_normal(vector, normal_traces):
    # if true, it is a normal behaviour
    return " ".join(vector) in normal_traces


if __name__ == "__main__":
    anom_types = ["all", "rework", "earlylate", "skip", "insert"]
    proportions = [5, 10, 15, 20, 30]
    
    for i in [1,2]: 
        for a in anom_types:
            for p in proportions:
                log_name = f"log{i}_anom_{a}_{p}"
                w2v_model = encode(log_name, min_count=1)
                train(log_name, f"normal_pn{i}", w2v_model)
