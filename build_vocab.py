import pickle
import argparse

START = "\\"


class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.char2idx = {}
        self.idx2char = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.char2idx:
            self.char2idx[word] = self.idx
            self.idx2char[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.char2idx:
            return self.char2idx['`']
        return self.char2idx[word]

    def __len__(self):
        return len(self.char2idx)

    def decode(self, idx):
        return self.idx2char[idx]

def build_vocab(caption_path):
    """Build a simple vocabulary wrapper."""
    with open(caption_path, "r") as f:
        text = f.read()
        vocab = sorted(set(text).union(set([START])))
    v = Vocabulary()
    v.char2idx = {u: i for i, u in enumerate(vocab)}
    v.idx2char = {i: u for i, u in enumerate(vocab)}

    return v


def main(args):
    vocab = build_vocab(args.caption_path)
    vocab_path = args.vocab_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str, 
                        default='data/captions_en5_preprocessed.txt',
                        help='path for train annotation file')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl', 
                        help='path for saving vocabulary wrapper')

    args = parser.parse_args()
    main(args)