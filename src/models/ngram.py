from ../utils import PREFIX_TOKEN, tokenize_line

class WordNGramLM:
    def __init__(self, N: int):
        if (N != int(N) or N < 1):
            raise ValueError("N should be a positive integer")
        self.N = N

    # for preprocessing an entire dataset
    def preprocess_sentences(sentences: List[str]): -> List[List[str]]
        return [preprocess_line(sentence) for sentence in sentences] 
    
    # for preprocessing a sample from a dataset
    def preprocess_line(line: str]): -> List[str]
        return [PREFIX_TOKEN * (self.N-1)] + tokenize_line(line) 
        

    def fit(self, train_data: List[str]):
        """
        Trains an N-gram language model.

        Inputs:
            - train_data: str, sentences in the training data

        """
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

    def eval_perplexity(self, eval_data: List[str]) -> float:

        """
        Evaluates the perplexity of the N-gram language model on the eval set.

        Input:
            - eval_data: List[str], the evaluation text

        Output:
            - float, the perplexity of the model on the evaluation set

        Note : For words that are not in the vocabulary, replace them with the <unk> token.
        Note : Don't count the <sos> tokens in your number of total tokens in order to match expected perplexities.

        """
        log_sum = 0
        total_words = 0
        eval_data = self.preprocess_data(eval_data, self.N)
        for words in eval_data:
          words = [word if word in self.vocabulary else "<unk>" for word in words]
          total_words += len(words) - (self.N - 1)
          for i in range(len(words) - self.N + 1):
            probability = self.get_probability_from_list(words[i:i+self.N])
            log_sum += np.log(probability)

        if total_words != 0:
          log_sum /= -total_words
        else:
          print("Error? No words in eval_data found")
        perplexity = np.exp(log_sum)
        return perplexity

    def sample_text(self, prefix: str = "<sos>", max_words: int = 100) -> str:

        """
        Samples text from the N-gram language model.
        Terminate sampling when either max_words is reached or when <eos> token is sampled.
        Inputs:
            - prefix: str, the prefix to start the sampling from. Can also be multiple words separated by spaces.
            - max_words: int, the maximum number of words to sample

        Outputs:
            - str, the sampled text

        Note: Please use np.random.choice for sampling next words
        """
        words = tokenize_line(prefix)
        k = max(self.N-1-len(words), 0)
        words = ["<sos>" for _ in range(k)] + words

        for _ in range(max_words):
          if words and words[-1] == "<eos>":
            break
          next_word = self.sample_next_word_list(words[-(self.N-1):])
          words.append(next_word)
        return " ".join(words[k:])

    # Extra utility functions
    
    # Increments to count corresponding to 'key' in count_dict
    def update_count(self, count_dict: Dict[str, int], key: str) -> None:
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

    # Given n tokens as 'ngramlist', returns the probability that
    # the nth token follows the previous n-1
    def get_probability_from_list(self, ngramlist: List[str]) -> float:
      ngram = " ".join(ngramlist)
      nminus1gram = " ".join(ngramlist[:-1])
      return self.get_probability(ngram, nminus1gram)

    # Samples the next word using the previous nminus1gram
    def sample_next_word(self, nminus1gram: str) -> str:
      if self.N == 1:
        return np.random.choice(self.vocablist, p=self.special_probs)
      elif nminus1gram not in self.nminus1grams_to_ngrams:
        return "<eos>"
      ngrams = list(self.nminus1grams_to_ngrams[nminus1gram])
      probs = np.array([self.get_probability(ngram, nminus1gram) for ngram in ngrams])
      probs /= np.sum(probs)
      chosen_ngram = np.random.choice(ngrams, p=probs)
      last_word_start = len(nminus1gram) + 1 if self.N != 1 else 0
      return chosen_ngram[last_word_start:]
        

    # Samples the next word using a list of the previous n-1 tokens
    def sample_next_word_list(self, nminus1gramlist: List[str]) -> str:
      nminus1gram = " ".join(nminus1gramlist)
      return self.sample_next_word(nminus1gram)
