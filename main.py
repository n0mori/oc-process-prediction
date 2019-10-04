#!/usr/bin/env python3.7

import encode
import svm
import argparse
import sys

def main(args):
    w2v_model = encode.train(args.log_name, min_count=1)
    svm.train(args.log_name, args.normal_traces, w2v_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log_name", help="name of the event log in the logs/ folder")
    parser.add_argument("normal_traces", help="name of the file containing the normal traces in the traces/ folder")

    args = parser.parse_args()
    main(args)
