#!/usr/bin/env python
import os
import string
import random
import pickle
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import defaultdict, Counter

class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """

    def __init__(self):
        self.N = 3 # n-gram size
        self.ngrams = defaultdict(Counter)
        self.PREFIX_CHAR = '|' # unlikely to show up in text

    @classmethod
    def load_training_data(cls):
        # TODO: replace with real HF data
        # For now, just dummy data
        return ["happy new year to you", "this is test data", "how are you today?", "happy new year", "that one is mine"]

    @classmethod
    def load_test_data(cls, fname):
        # your code here
        data = []
        with open(fname) as f:
            for line in f:
                inp = line[:-1]  # the last character is a newline
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def preprocess_for_ngram(self, data):
        return [(self.PREFIX_CHAR * self.N) + line.lower() for line in data]

    def run_train(self, data, work_dir):
        # Build up n-grams from training data
        data = self.preprocess_for_ngram(data)
        for line in data:
            for i in range(len(line) - self.N):
                context = line[i:i+self.N]
                next_char = line[i+self.N]
                self.ngrams[context][next_char] += 1

    def run_pred(self, data):
        # your code here
        data = self.preprocess_for_ngram(data)
        preds = []
        all_chars = string.ascii_letters
        for line in data:
            context = line[-self.N:]
            curr_preds = self.ngrams.get(context, Counter())
            curr_preds = [char for char, _ in curr_preds.most_common(3)]
            
            # ensure always at least 3 choices without repeats
            if len(curr_preds) != 3:
                random_preds = random.sample(all_chars, k=3+len(curr_preds))
                random_preds = [pred for pred in random_preds if pred not in curr_preds]
                curr_preds = (curr_preds + random_preds)[:3]
            preds.append(''.join(curr_preds))
        return preds

    def save(self, work_dir):
        # Save the trained ngrams to a checkpoint file
        with open(os.path.join(work_dir, 'model.checkpoint'), 'wb') as f:
            pickle.dump(self.ngrams, f)

    @classmethod
    def load(cls, work_dir):
        # Load the trained ngrams from checkpoint
        model = MyModel()
        with open(os.path.join(work_dir, 'model.checkpoint'), 'rb') as f:
            model.ngrams = pickle.load(f)
        return model


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instatiating model')
        model = MyModel()
        print('Loading training data')
        train_data = MyModel.load_training_data()
        print('Training')
        model.run_train(train_data, args.work_dir)
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(args.work_dir)
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
