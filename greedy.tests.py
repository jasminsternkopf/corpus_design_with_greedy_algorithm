import unittest

import numpy as np

from greedy import (count_occurrences_of_unit_in_list_of_sentences,
                    get_distribution, greedy, kullback_leibler_div)


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

  # region kullback_leibler_div

  def test_kullback_leiber(self):
    dist_1 = {("Hallo", "h"): 1 / 7, ("du", "d"): 3 / 7,
              ("und", "u"): 2 / 7, ("Bye", "b"): 1 / 7}
    dist_2 = {("Hallo", "h"): 0.2, ("du", "d"): 0.3, ("und", "u"): 0.4, ("Bye", "b"): 0.1}
    res = kullback_leibler_div(dist_1, dist_2)
    right_div = 1 / 7 * np.log((1 / 7) / 0.2) + 3 / 7 * np.log((3 / 7) / 0.3) + \
        2 / 7 * np.log((2 / 7) / 0.4) + 1 / 7 * np.log((1 / 7) / 0.1)

    self.assertAlmostEqual(right_div, res)

  def test_kullback_leiber__same_dist__expect_zero(self):
    dist_1 = {("Hallo", "h"): 1 / 7, ("du", "d"): 3 / 7,
              ("und", "u"): 2 / 7, ("Bye", "b"): 1 / 7}
    res = kullback_leibler_div(dist_1, dist_1)

    self.assertEqual(0, res)

  # endregion

  # region greedy

  def test_greedy__one_iteration(self):
    target_dist = {("Hallo", "h"): 0.2, ("du", "d"): 0.3, ("und", "u"): 0.4, ("Bye", "b"): 0.1}
    selection_list = [[("Hallo", "h"), ("du", "d"), ("und", "u"), ("du", "d")],
                      [("Bye", "b"), ("und", "u"), ("du", "d")],
                      [("Irrelevante", "i"), ("Worte", "w")]]
    res = greedy(target_dist, selection_list, 1)
    right_cover = [[("Hallo", "h"), ("du", "d"), ("und", "u"), ("du", "d")]]

    self.assertEqual(1, len(res))
    self.assertEqual(right_cover, res)

  def test_greedy__two_iterations(self):
    target_dist = {("Hallo", "h"): 0.2, ("du", "d"): 0.3, ("und", "u"): 0.4, ("Bye", "b"): 0.1}
    selection_list = [[("Hallo", "h"), ("du", "d"), ("und", "u"), ("du", "d")],
                      [("Bye", "b"), ("und", "u"), ("du", "d")],
                      [("Irrelevante", "i"), ("Worte", "w")]]
    res = greedy(target_dist, selection_list, 2)
    right_cover = [[("Hallo", "h"), ("du", "d"), ("und", "u"), ("du", "d")],
                   [("Bye", "b"), ("und", "u"), ("du", "d")]]

    self.assertEqual(2, len(res))
    self.assertEqual(right_cover, res)

  def test_greedy__three_iterations(self):
    target_dist = {("Hallo", "h"): 0.2, ("du", "d"): 0.3, ("und", "u"): 0.4, ("Bye", "b"): 0.1}
    selection_list = [[("Hallo", "h"), ("du", "d"), ("und", "u"), ("du", "d")],
                      [("Bye", "b"), ("und", "u"), ("du", "d")],
                      [("Irrelevante", "i"), ("Worte", "w")]]
    res = greedy(target_dist, selection_list, 3)
    right_cover = [[("Hallo", "h"), ("du", "d"), ("und", "u"), ("du", "d")],
                   [("Bye", "b"), ("und", "u"), ("du", "d")],
                   [("Irrelevante", "i"), ("Worte", "w")]]

    self.assertEqual(3, len(res))
    self.assertEqual(right_cover, res)

  # endregion


if __name__ == '__main__':
  suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
  unittest.TextTestRunner(verbosity=2).run(suite)
