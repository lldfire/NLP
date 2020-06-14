from src.hmm import train_hmm, use_cut


__all__ = ['train', 'cut']     # 限定被导入时只能使用的变量


def train():
    train_hmm().train()


def cut(text):
    print(use_cut().cut(text))
