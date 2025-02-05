import argparse
import re


def parse_number_list(value):
    matched = re.match(r'\[(.*)\]', value)
    if not matched:
        raise argparse.ArgumentTypeError("Invalid format for list.")

    number_str = matched.group(1)
    numbers = [int(v.strip()) for v in number_str.split(',') if v.strip().isdigit()]
    return numbers


parser = argparse.ArgumentParser(description='argparser.')

parser.add_argument('--project', type=str, required=False, help='Project name of run.')
parser.add_argument('--name', type=str, required=False, help='Name of run.')
parser.add_argument('--dataset', type=str, required=False, help='Path of dataset.')
parser.add_argument('--output', type=str, required=False, help='Output dir of logs and weights.')
parser.add_argument('--cut', type=float, help='Split rate of test set.')
parser.add_argument('--device', type=parse_number_list, help='Device id list of cuda.')
parser.add_argument('--batch-size', type=int, help='Batch size of data.')
parser.add_argument('--epoch', type=int, help='Epoch of train.')
parser.add_argument('--lr', type=float, help='Learning rate of model.')
parser.add_argument('--dropout', type=float, help='Dropout of model.')
parser.add_argument('--loss-type', type=str, help='Loss type of train.')

args = parser.parse_args()
