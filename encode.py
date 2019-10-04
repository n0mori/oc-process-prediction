#!/usr/bin/env python3.7

import os
import re
from reader import get_traces
from gensim.models import Word2Vec
import numpy as np


def train(filename, min_count=1):
    traces = get_traces("logs/" + filename + ".csv")
    model = Word2Vec(traces, min_count=1, workers=8)
    model = Word2Vec(sentences=traces, min_count=min_count)
    return model


def train_batch():
    for filenames in os.listdir("logs"):
        if re.search(r"\.csv", filenames):
            train(filenames)


def create_vectors(model, vectors):
    trace_vectors = []
    for trace in vectors:
        v = np.array(model.wv[trace[0]])

        for word in trace[1:]:
            v = list(map(sum, zip(v, model.wv[word])))
        vec = np.array(v)
        vec = vec / len(trace)
        trace_vectors.append(v)
    return trace_vectors


if __name__ == "__main__":
    train_batch()
