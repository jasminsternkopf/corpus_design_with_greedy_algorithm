import unittest

from greedy import (count_occurrences_of_unit_in_list_of_sentences,
                    find_sentence_with_min_div, get_distribution,
                    get_divergence_for_sentence, greedy)


class UnitTests(unittest.TestCase):
  def __init__(self, methodName: str) -> None:
    super().__init__(methodName)

  # region count_occurrences_of_unit_in_list_of_sentences

  def test_count_occurrences_of_unit_in_list_of_sentences_du_3(self):
    test_list = [[("Hallo", "h"), ("du", "d"), ("und", "u"), ("du", "d")],
                 [("Bye", "b"), ("und", "u"), ("du", "d")]]
    res = count_occurrences_of_unit_in_list_of_sentences(("du", "d"), test_list)

    self.assertEqual(3, res)

  def test_count_occurrences_of_unit_in_list_of_sentences_Hallo_1(self):
    test_list = [[("Hallo", "h"), ("du", "d"), ("und", "u"), ("du", "d")],
                 [("Bye", "b"), ("und", "u"), ("du", "d")]]
    res = count_occurrences_of_unit_in_list_of_sentences(("Hallo", "h"), test_list)

    self.assertEqual(1, res)

  # endregion

  # region get_distribuion

  def test_get_distribution(self):
    test_list = [[("Hallo", "h"), ("du", "d"), ("und", "u"), ("du", "d")],
                 [("Bye", "b"), ("und", "u"), ("du", "d")]]
    target_dist = {("Hallo", "h"): 0.2, ("du", "d"): 0.3, ("und", "u"): 0.4, ("Bye", "b"): 0.1}
    res = get_distribution(test_list, target_dist)
    right_dist = {("Hallo", "h"): 1 / 7, ("du", "d"): 3 / 7,
                  ("und", "u"): 2 / 7, ("Bye", "b"): 1 / 7}

    self.assertEqual(len(res), len(right_dist))

    for key in res.keys():
      self.assertAlmostEqual(res[key], right_dist[key])

  # endregion

  # region get_divergence_for_sentence
if __name__ == '__main__':
  suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
  unittest.TextTestRunner(verbosity=2).run(suite)
