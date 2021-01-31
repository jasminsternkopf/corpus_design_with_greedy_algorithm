from collections import Counter
from typing import Dict, List, Tuple

import numpy as np


def greedy(target_dist: Dict[Tuple[str, str], float], selection_list: List[List[Tuple[str, str]]], max_iter: int) -> List[List[Tuple[str, str]]]:
  cover: List[List[Tuple[str, str]]] = []
  for _ in range(max_iter):
    best_sentence = find_sentence_with_min_div(selection_list, cover, target_dist)
    cover.append(best_sentence)
    selection_list.remove(best_sentence)
  return cover


def find_sentence_with_min_div(selection_list, cover, target_dist) -> List[Tuple[str, str]]:
  min_div = get_divergence_for_sentence(selection_list[0], cover, target_dist)
  best_sentence = selection_list[0]
  for sentence in selection_list[1:]:
    if get_divergence_for_sentence(sentence, cover, target_dist) < min_div:
      best_sentence = sentence
  return best_sentence


def get_divergence_for_sentence(sentence, cover, target_dist) -> float:
  cover = cover.append(sentence)
  cover_dist = get_distribution(cover, target_dist)
  divergence = kullback_leibler_div(cover_dist, target_dist)
  return divergence


def kullback_leibler_div(dist_1: Dict[Tuple[str, str], float], dist_2: Dict[Tuple[str, str], float]) -> float:
  for value in dist_2.values():
    assert value > 0
  unequal_zero_keys = [key for key in dist_1.keys() if dist_1[key] > 0]
  divergence = [dist_1[key] * (np.log(dist_1[key]) - np.log(dist_2[key]))
                for key in unequal_zero_keys]
  return sum(divergence)


def get_distribution(sentence_list: List[List[Tuple[str, str]]], target_dist: Dict[Tuple[str, str], float]) -> Dict[Tuple[str, str], float]:
  new_dist = {key: count_occurrences_of_unit_in_list_of_sentences(
    key, sentence_list) for key in target_dist.keys()}
  total_number_of_single_units = sum(new_dist.values())
  for key, value in new_dist.items():
    new_dist[key] = value / total_number_of_single_units
  return new_dist


def count_occurrences_of_unit_in_list_of_sentences(unit: Tuple[str, str], sentence_list: List[List[Tuple[str, str]]]) -> int:
  occurence_list = [Counter(sentence)[unit] for sentence in sentence_list]
  return sum(occurence_list)
