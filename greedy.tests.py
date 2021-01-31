from greedy import greedy, find_sentence_with_min_div, get_distribution, get_divergence_for_sentence, count_occurrences_of_unit_in_list_of_sentences, count_occurrences_of_unit_in_one_sentence
import unittest

class UnitTests(unittest.TestCase):
  def __init__(self, methodName: str) -> None:
    super().__init__(methodName)