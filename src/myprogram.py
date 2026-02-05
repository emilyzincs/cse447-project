#!/usr/bin/env python
import os
import string
import random
import pickle
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import defaultdict, Counter
from utils import *
import numpy as np

class MyModel:
    def __init__(self):
        self.N = 3


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


    # for preprocessing an entire dataset
    def preprocess_data(self, sentences):
        return [self.preprocess_line(sentence) for sentence in sentences] 
    
    # for preprocessing a sample from a dataset
    def preprocess_line(self, line):
        return [PREFIX_TOKEN for _ in range(self.N-1)] + tokenize_line(line) 

    # Build up n-grams from training data
    def run_train(self, train_data, work_dir):
        self.ngram_counts = {}
        self.nminus1gram_counts = {}
        self.nminus1grams_to_ngrams = {}
        self.vocabulary = {}
        self.corpus_word_count = 0
        train_data = self.preprocess_data(train_data)
        for words in train_data:
          for word in words:
            if word not in self.vocabulary:
              self.vocabulary[word] = 1
            else:
              self.vocabulary[word] += 1
          self.corpus_word_count += len(words) - (self.N-1)
          for i in range(len(words) - self.N + 1):
            nminus1gram = " ".join(words[i:i+self.N-1])
            self.update_count(self.nminus1gram_counts, nminus1gram)
            ngram = " ".join(words[i:i+self.N])
            self.update_count(self.ngram_counts, ngram)
            if nminus1gram not in self.nminus1grams_to_ngrams:
              self.nminus1grams_to_ngrams[nminus1gram] = set()
            self.nminus1grams_to_ngrams[nminus1gram].add(ngram)

        self.vocablist = list(self.vocabulary)
        self.vocablist = [word for word in self.vocablist if word != PREFIX_TOKEN]
        if self.N == 1:
          self.special_probs = [self.vocabulary[word] / self.corpus_word_count for word in self.vocablist]

    # Gets the ngrams associated with the give nminus1gram
    def get_ngrams(self, nminus1gram):
        if self.N == 1:
            return self.vocablist
        return self.nminus1grams_to_ngrams.get(nminus1gram, set())

    def run_pred(self, data):
        data = self.preprocess_data(data)
        preds = []
        all_chars = string.ascii_letters
        for words in data:
            candidates_to_probs = defaultdict(float)
            p_same_word = 0

            if len(words) >= self.N:
                curr_nminus1gram = " ".join(words[-self.N:-1])
                last_word_start = (len(curr_nminus1gram) + 1) if self.N != 1 else 0
                prefix = words[-1]
                curr_ngrams = self.get_ngrams(curr_nminus1gram)
                curr_ngrams = filter_prefix(prefix, last_word_start, curr_ngrams)
                candidate_position = last_word_start + len(prefix)
                for ngram in curr_ngrams:
                    p = self.get_probability(ngram, curr_nminus1gram)
                    candidates_to_probs[ngram[candidate_position]] += p
                    p_same_word += p
                p_same_word = min(p_same_word, 1)

            p_diff_word = 1 - p_same_word
            next_nminus1gram = " ".join(words[-(self.N-1):])
            last_char_lang = get_lang(words[-1][-1])
            if last_char_lang not in NON_SPACE_DELIMITED_LANGS:
                candidates_to_probs[' '] += p_diff_word
            elif next_nminus1gram in self.nminus1gram_counts:
                next_ngrams = self.get_ngrams(next_nminus1gram)
                last_word_start = (len(next_nminus1gram) + 1) if self.N != 1 else 0
                for ngram in next_ngrams:
                    p = p_diff_word * self.get_probability(ngram, next_nminus1gram)
                    ch = ngram[last_word_start]
                    if get_lang(ch) != last_char_lang:
                        candidates_to_probs[' '] += p
                    else:
                        candidates_to_probs[ch] += p

            candidates, probs = zip(*candidates_to_probs.items())
            probs = np.array(probs, dtype=float)
            total = np.sum(probs)
            if (total > 0):
                probs /= total
            top_idxs = np.argsort(probs)[-CHARS_TO_PREDICT:][::-1]
            curr_preds = [candidates[i] for i in top_idxs]
            
            # ensure always at least 'CHARS_TO_PREDUCT' choices without repeats
            if len(curr_preds) != CHARS_TO_PREDICT:
                random_preds = random.sample(all_chars, k=CHARS_TO_PREDICT+len(curr_preds))
                random_preds = [pred for pred in random_preds if pred not in curr_preds]
                curr_preds = (curr_preds + random_preds)[:CHARS_TO_PREDICT]
            preds.append(''.join(curr_preds))
        return preds

    # Increments the count corresponding to 'key' in count_dict
    def update_count(self, count_dict: dict[str, int], key: str) -> None:
      if key in count_dict:
        count_dict[key] += 1
      else:
        count_dict[key] = 1

    # Returns the probability of the 'ngram' following the 'nminus1gram'
    def get_probability(self, ngram: str, nminus1gram: str) -> float:
      probability = 0
      if ngram in self.ngram_counts:
        probability = self.ngram_counts[ngram] / (self.nminus1gram_counts[nminus1gram] if self.N != 1 else self.corpus_word_count)
      return probability

    def save(self, work_dir):
        # Save the trained model to a checkpoint file
        with open(os.path.join(work_dir, 'model.checkpoint'), 'wb') as f:
            pickle.dump(self.ngram_counts, f)
            pickle.dump(self.nminus1gram_counts, f)
            pickle.dump(self.nminus1grams_to_ngrams, f)
            pickle.dump(self.vocabulary, f)
            pickle.dump(self.corpus_word_count, f)
            pickle.dump(self.vocablist, f)
            if self.N == 1:
                pickle.dump(self.special_probs, f)

    @classmethod
    def load(cls, work_dir):
        # Load the trained ngrams from checkpoint
        model = MyModel()
        with open(os.path.join(work_dir, 'model.checkpoint'), 'rb') as f:
            model.ngram_counts = pickle.load(f)
            model.nminus1gram_counts = pickle.load(f)
            model.nminus1grams_to_ngrams = pickle.load(f)
            model.vocabulary = pickle.load(f)
            model.corpus_word_count = pickle.load(f)
            model.vocablist = pickle.load(f)
            if model.N == 1:
                model.special_probs = pickle.load(f)
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
