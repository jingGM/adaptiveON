import argparse
import numpy as np
from typing import Tuple, List


def get_args():
    parser = argparse.ArgumentParser(description='Adaptive RL')
    parser.add_argument('-l', '--list', nargs='+', help='<Required> Set flag', required=True, type=float)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    values = np.array(args.list)
    print(values)
