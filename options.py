import argparse

parser = argparse.ArgumentParser()
parser.add_argument('experiment_name', type=str, help='Experiment_name')
parser.add_argument('--model', default='crnn', type=str, choices=['crnn'])
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--bs', default=16, type=int, help='Batch size')
parser.add_argument('--lr', default=0.01, type=float, help='Learning rate')
cfg = parser.parse_args()
