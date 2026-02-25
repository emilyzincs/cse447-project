#!/usr/bin/env python
import os
import sys
import random
import pickle
import traceback
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import defaultdict, Counter
from datasets import load_dataset
from itertools import islice
from utils import *

class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """

    def __init__(self):
        self.N = 3 # n-gram size
        self.ngrams = defaultdict(Counter)
        self.PREFIX_CHAR = '|' # unlikely to show up in text

    @classmethod
    def load_training_data(cls, languages, max_docs_per_lang, streaming):
        """
        Load training data from HuggingFace Cohere Wikipedia dataset.
        
        Args:
            languages: List of language codes to load (default: ['simple', 'russian', 'chinese_simplified'])
            max_docs_per_lang: Maximum number of documents to load per language
            streaming: Whether to stream the dataset (True) or download it (False)
        
        Returns:
            List of text documents
        """        
        data = []
        dataset_name = "Cohere/wikipedia-2023-11-embed-multilingual-v3" # from hf
        
        for lang in languages:
            docs_loaded = 0
            print(f"Loading {lang} from HuggingFace dataset...")
            try:
                docs_stream = load_dataset(
                    dataset_name,
                    lang,
                    split="train",
                    streaming=streaming
                )
                
                for doc in islice(docs_stream, max_docs_per_lang):
                    # Extract text from the document
                    text = doc.get('text', '')
                    if text:
                        data.append(text)
                        docs_loaded += 1
                        if len(data) % 5000 == 0:
                            print(f"Loaded {len(data)} total documents so far") 
                    if docs_loaded >= max_docs_per_lang:
                        break

            except Exception as e:
                print(f"Warning: Could not load {lang}: {e}")
        
        print(f"Total documents loaded: {len(data)}")
        return data

    @classmethod
    def load_test_data(cls, fname):
        # your code here
        data = []
        with open(fname) as f:
            for line in f:
                if len(line) > 0:
                    inp = line[:-1]  # the last character is a newline
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def preprocess_data(self, data):
        return [self.preprocess_line(line) for line in data]
    
    def preprocess_line(self, line):
        return (self.PREFIX_CHAR * self.N) + line.lower()
        

    def run_train(self, data, work_dir):
        # Build up n-grams from training data
        data = self.preprocess_data(data)
        for line in data:
            for i in range(len(line) - self.N):
                context = line[i:i+self.N]
                next_char = line[i+self.N]
                self.ngrams[context][next_char] += 1

    def run_pred(self, data):
        # your code here
        preds = []
        all_chars = string.ascii_letters
        print("num data: ", len(data))        
        for i,unprocessed_line in enumerate(data):
            curr_preds = []
            try:
                line = self.preprocess_line(unprocessed_line)
                if len(line) >= self.N:
                    context = line[-self.N:]
                    curr_preds = self.ngrams.get(context, Counter())
                    curr_preds = [char for char, _ in curr_preds.most_common(3)]
            except Exception as e:
                print(f"Exception at line {i} of pred data:")
                print(e)
                traceback.print_exc()
            finally:
                # ensure always at least 3 choices without repeats
                if len(curr_preds) != 3:
                    lang = get_last_lang(unprocessed_line)
                    all_chars = None
                    if lang == ENG:
                        all_chars = ENG_CHARS
                    elif lang == RUS:
                        all_chars = RUS_CHARS
                    elif lang == CHI:
                        all_chars = CHI_CHARS
                    else:
                        all_chars = ENG_CHARS # just use latin characters as default for UNK langs
                    random_preds = random.sample(all_chars, k=CHARS_TO_PREDICT+len(curr_preds))
                    random_preds = [pred for pred in random_preds if pred not in curr_preds]
                    curr_preds = (curr_preds + random_preds)[:CHARS_TO_PREDICT]
                preds.append(''.join(curr_preds))
        return preds

    def save(self, work_dir):
        # Save the trained ngrams to a checkpoint file
        with open(os.path.join(work_dir, 'model.checkpoint'), 'wb') as f:
            size = sys.getsizeof(pickle.dumps(self.ngrams))
            print(f"Approximate pickle size: {size / 1e6:.2f} MB")
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
    parser.add_argument('--languages', nargs='+', help='languages to load from HF dataset', 
                        default=['en', 'ru', 'zh'])
    parser.add_argument('--max_docs', type=int, help='max documents per language', default=100)
    parser.add_argument('--stream', action='store_true', help='stream dataset instead of downloading', 
                        default=True)
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instantiating model')
        model = MyModel()
        print('Loading training data from HuggingFace dataset')
        train_data = MyModel.load_training_data(
            languages=args.languages,
            max_docs_per_lang=args.max_docs,
            streaming=args.stream,
        )
        print('Training')
        model.run_train(train_data, args.work_dir)
        print('Saving model')
        model.save(args.work_dir)
        # os._exit(0)
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
