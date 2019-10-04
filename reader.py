#!/usr/bin/env python3.7

import csv


def get_traces(filename):
    traces = {}
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        lst = list(reader)
        for row in lst[1:]:
            if row[0] not in traces:
                traces[row[0]] = []
            traces[row[0]].append(row[1:])

    phrases = []
    for v in traces.values():
        phrase = [trace[0] for trace in v]
        phrases.append(phrase)

    return phrases
