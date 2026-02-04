from .ngram import WordNGramLM

class WordNGramLMWithInterpolation(WordNGramLM):
    """
    Remember you can use the inheritance from WordNGramLM in your implementation!
    """

    def __init__(self, N: int, lambdas: List[float]):

        """
        Constructor for WordNGramLMWithInterpolation class.
        Inputs:
            - N: int, the N in N-gram
            - lambdas: List[float], the list of lambdas for interpolation between 1-gram, 2-gram, 3-gram, ..., N-gram models
                Note: The length of lambdas should be N. The sum of lambdas should be 1. lambdas[0] corresponds to 1-gram model, lambdas[1] corresponds to 2-gram model and so on.
        """
        if len(lambdas) != N:
            raise ValueError("Length of lambdas should be equal to N")
        if abs(sum(lambdas) - 1) >= 0.0001:
            raise ValueError("Sum of lambdas should be equal to 1")
        super().__init__(N)
        self.lambdas = lambdas


    def fit(self, train_data: List[str]):

        """
        Trains an N-gram language model with interpolation.

        Inputs:
            - train_data: str, sentences in the training data

        """
        self.models = []
        for i in range(1, self.N+1):
          model = WordNGramLM(i)
          model.fit(train_data)
          self.models.append(model)
        self.vocabulary = self.models[0].vocabulary
        self.vocablist = self.models[0].vocablist
        self.corpus_word_count = self.models[0].corpus_word_count
        self.preweighted_unigram_probs = {word: self.lambdas[0] * self.vocabulary[word] / self.corpus_word_count for word in self.vocabulary}

    def eval_perplexity(self, eval_data: List[str]) -> float:
        """
        Evaluates the perplexity of the N-gram language model with interpolation on the eval set.

        Input:
            - eval_data: List[str], the evaluation text

        Output:
            - float, the perplexity of the model on the evaluation set

        Note : For tokens that are not in the vocabulary, replace them with the <unk> token.

        """
        return super().eval_perplexity(eval_data)

    def sample_text(self, prefix: str = "<sos>", max_words: int = 100) -> float:

        """
        Samples text from the N-gram language model with interpolation.

        Inputs:
            - prefix: str, the prefix to start the sampling from. Can also be multiple words separated by spaces.
            - max_words: int, the maximum number of words to sample

        Outputs:
            - str, the sampled text

        Note: Please use np.random.choice for sampling next words
        """
        return super().sample_text(prefix, max_words)

    # Extra utility functions

    # Given n tokens as 'ngramlist', returns the probability that
    # the nth token follows the previous n-1
    def get_probability_from_list(self, ngramlist: List[str]) -> float:
      N = len(ngramlist)
      ret = 0
      if N != 1:
        ret += self.get_probability_from_list(ngramlist[1:])
      return ret + self.lambdas[N-1] * self.models[N-1].get_probability_from_list(ngramlist)

    # Samples the next word using the previous nminus1gram
    def sample_next_word_list(self, nminus1gramlist: List[str]) -> str:
      prob_map = self.preweighted_unigram_probs.copy()
      for i in range(1, self.N):
        model = self.models[i]
        nminus1gram = " ".join(nminus1gramlist[-i:])
        last_word_start = len(nminus1gram) + 1
        ngrams = model.nminus1grams_to_ngrams.get(nminus1gram, None)
        for ngram in ngrams:
            prob = self.lambdas[i] * model.get_probability(ngram, nminus1gram)
            word = ngram[last_word_start:]
            prob_map[word] += prob
      probs = np.array([prob_map[word] for word in self.vocablist])
      probs = probs / np.sum(probs)
      return np.random.choice(self.vocablist, p=probs)
