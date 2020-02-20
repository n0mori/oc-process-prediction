#!/usr/bin/env python3.7

import csv


def get_traces(filename):
    traces = {}
    idx_label = 0
    idx_activity = 0
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        lst = list(reader)
        idx_label = lst[0].index('label')
        idx_activity = lst[0].index('activity_name')
        for row in lst[1:]:
            if row[0] not in traces:
                traces[row[0]] = []
            traces[row[0]].append(row)

    phrases = []
    labels = []
    for v in traces.values():
        label = 1 if v[0][idx_label] == 'normal' else -1
        phrase = [trace[idx_activity].replace(' ', '') for trace in v]
        phrases.append(phrase)
        labels.append(label)

    return phrases, labels
