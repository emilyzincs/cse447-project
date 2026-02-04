#!/usr/bin/env python
import os
import string
import random
import pickle
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import defaultdict, Counter
from datasets import load_dataset

class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """

    def __init__(self):
        self.N = 3 # n-gram size
        self.ngrams = defaultdict(Counter)
        self.PREFIX_CHAR = '|' # unlikely to show up in text

    @classmethod
    def load_training_data(cls, languages=None, max_docs_per_lang=1000, streaming=True):
        """
        Load training data from HuggingFace Cohere Wikipedia dataset.
        
        Args:
            languages: List of language codes to load (default: ['simple', 'russian', 'chinese_simplified'])
            max_docs_per_lang: Maximum number of documents to load per language
            streaming: Whether to stream the dataset (True) or download it (False)
        
        Returns:
            List of text documents
        """
        if languages is None:
            languages = ['simple', 'russian', 'chinese_simplified']
        
        data = []
        dataset_name = "Cohere/wikipedia-2023-11-embed-multilingual-v3" # from hf
        
        for lang in languages:
            print(f"Loading {lang} from HuggingFace dataset...")
            try:
                docs_stream = load_dataset(
                    dataset_name,
                    lang,
                    split="train",
                    streaming=streaming
                )
                
                for doc in docs_stream:
                    # Extract text from the document
                    text = doc.get('text', '')
                    if text:
                        data.append(text)
                    
                    if len(data) >= max_docs_per_lang * len(languages):
                        break
                
                print(f"Loaded {len(data)} total documents so far")
            except Exception as e:
                print(f"Warning: Could not load {lang}: {e}")
        
        print(f"Total documents loaded: {len(data)}")
        return data

    @classmethod
    def load_test_data(cls, fname, min_length=1):
        """
        Load test data from file.
        
        Args:
            fname: Path to test data file
            min_length: Minimum length of text to include (filters very short lines)
        
        Returns:
            List of text samples
        """
        data = []
        try:
            with open(fname, 'r', encoding='utf-8') as f:
                for line in f:
                    # Strip whitespace and newlines
                    inp = line.strip()
                    
                    # Skip empty lines
                    if len(inp) >= min_length:
                        data.append(inp)
            
            print(f"Loaded {len(data)} test samples from {fname}")
        except FileNotFoundError:
            print(f"Error: Test file {fname} not found")
            return []
        except Exception as e:
            print(f"Error reading test file: {e}")
            return []
        
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
    parser.add_argument('--languages', nargs='+', help='languages to load from HF dataset', 
                        default=['simple', 'ru', 'zh'])
    parser.add_argument('--max_docs', type=int, help='max documents per language', default=1000)
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
            streaming=args.stream
        )
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
