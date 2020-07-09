import sys
from src.api import train, cut


def manage():
    args = sys.argv[1]
    if args == 'train':
        train()
    elif args == 'cut':
        cut(sys.argv[2])


if __name__ == '__main__':
    manage()
