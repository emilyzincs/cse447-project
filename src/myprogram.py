#!/usr/bin/env python
import os
import string
import random
import pickle
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import defaultdict, Counter
from ./models import WordNGramLM
from utils import CHARS_TO_PREDICT

class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """

    def __init__(self):
        self.model = WordNGramLM(3)


    @classmethod
    def load_training_data(cls):
        # TODO: replace with real HF data
        # For now, just dummy data
        return SAMPLE_ENGLISH_DATA


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


    def run_train(self, data, work_dir):
        # Build up n-grams from training data
        self.model.fit(data)

    def run_pred(self, data):
        # your code here
        data = self.model.preprocess_data(data)
        preds = []
        all_chars = string.ascii_letters
        for words in data:
            curr_preds = []
            if len_words >= self.model.N:
                # For now just hardcoding since couldn't think of an elegant workaround
                probability_new_word = (
                    self.model.get_probability_from_list(words[-self.model.N:]) if data else 0
                )
                prefix = data[-1]
                nminus1gram = data[-N:-1]
                ngrams = self.model.vocablist if self.model.N == 1 else self.model.nminus1grams_to_ngrams[nminus1gram]
                last_word_start = len(nminus1gram) + 1 if self.model.N != 1 else 0
                ngrams = [
                    ngram if (
                        len(ngram) > last_word_start + len(prefix) and
                        ngram[last_word_start:last_word_start + len(prefix)] == prefix
                    ) for ngram in ngrams
                ]
                candidates_to_probs = defaultdict(int)
                for ngram in ngrams:
                    candidates_to_probs[ngram[last_word_start + len(prefix)]] += self.model.get_probability(ngram, nminus1gram)
                candidates, probs = zip(*candidates_to_probs.items())
                probs = np.array(probs)
                probs /= np.sum(probs)
                if (probability_new_word != 0):
                    probs *= 1 - probability_new_word
                    candidates.append(' ')
                    probs.append(probability_new_word)
                curr_preds = np.random.choice(candidates, size=max(len(candidates), CHARS_TO_PREDICT), replace=False, p=probs)
            
            # ensure always at least 3 choices without repeats
            if len(curr_preds) != CHARS_TO_PREDICT:
                random_preds = random.sample(all_chars, k=CHARS_TO_PREDICT+len(curr_preds))
                random_preds = [pred for pred in random_preds if pred not in curr_preds]
                curr_preds = (curr_preds + random_preds)[:CHARS_TO_PREDICT]
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
