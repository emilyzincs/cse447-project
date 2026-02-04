from .ngram import WordNGramLM

class WordNGramLMWithAddKSmoothing(WordNGramLM):
    """
    Remember you can use the inheritance from WordNGramLM in your implementation!
    """

    def __init__(self, N: int, k: int = 1):
        super().__init__(N)
        self.k = k

    def fit(self, train_data: List[str]):
        """
        Trains an N-gram language model with Add-k smoothing.

        Inputs:
            - train_data: str, sentences in the training data

        """
        super().fit(train_data)
        self.vocab = self.vocablist # gradescope test expects this field ?
        if self.N == 1:
          self.special_probs = [(self.vocabulary[word] + self.k) / (self.corpus_word_count + self.k * len(self.vocablist)) for word in self.vocablist]


    def eval_perplexity(self, eval_data: List[str]) -> float:
        """
        Evaluates the perplexity of the N-gram language model with Add-k smoothing on the eval set.

        Input:
            - eval_data: List[str], the evaluation text

        Output:
            - float, the perplexity of the model on the evaluation set

        Note : For tokens that are not in the vocabulary, replace them with the <unk> token.

        """
        return super().eval_perplexity(eval_data)

    def sample_text(
        self, prefix: str = "<sos>", max_words: int = 100,
    ) -> float:
        """
        Samples text from the N-gram language model.

        Inputs:
            - prefix: str, the prefix to start the sampling from. Can also be multiple words separated by spaces.
            - max_words: int, the maximum number of words to sample

        Outputs:
            - str, the sampled text

        Note: Please use np.random.choice for sampling next words
        """
        words = prefix.split()
        m = max(self.N-1-len(words), 0)
        words = ["<sos>" for _ in range(m)] + words
        for _ in range(max_words):
          if words and words[-1] == "<eos>":
            break
          next_word = None

          if self.N == 1:
            next_word = np.random.choice(self.vocablist, p=self.special_probs)
          else:
            nminus1list = words[-(self.N-1):]
            nminus1gram = " ".join(words[-(self.N-1):])
            probabilities = []
            for word in self.vocablist:
              nminus1list.append(word)
              ngram = " ".join(nminus1list)
              probabilities.append(self.get_probability(ngram, nminus1gram))
              nminus1list.pop()
            probabilities = np.array(probabilities)
            probabilities /= np.sum(probabilities)
            next_word = np.random.choice(self.vocablist, p=probabilities)
          words.append(next_word)
        return " ".join(words)

    # Extra utility functions
    # Returns the probability of the given 'ngram' being formed given the
    # previous 'nminus1gram'
    def get_probability(self, ngram: str, nminus1gram: str) -> float:
      numerator = self.ngram_counts.get(ngram, 0) + self.k
      denominator = self.nminus1gram_counts.get(nminus1gram, 0) + self.k * len(self.vocabulary)
      return numerator / denominator
