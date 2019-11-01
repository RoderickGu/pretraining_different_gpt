import sys
from utils import parse_args
import torch

if __name__ == '__main__':
    args = parse_args()

    print(args.local_rank)

    breakpoint()
