# -*- coding: utf-8 -*-
__author__ = 'Lara Olmos Camarena'


import os
import numpy as np
import itertools

from extractor import *
from preprocessor import *



class STS_Corpus():

	def __init__(self):
		self.input_path = os.path.join(os.getcwd(), 'dataset')
		self.text_extractor = STSExtractor()
		self.text_tokenizer = TextTokenizer()
		self.text_normalizer = TextNormalizer()
		self.examples, self.processed_text, self.partitions_normalized, self.terms, self.sts_dataset = self._build()

	def _build(self):
		dir_list_files = sorted(os.listdir(self.input_path))
		examples, pearsons = {}, {}
		processed_text, sts_dataset = [], []
		partitions, partitions_normalized = {}, {}
		terms = {}
		for file_name in dir_list_files:
			examples[file_name] = self.text_extractor.extract_file_examples(os.path.join(self.input_path, file_name))
			sts_dataset += [(self.text_normalizer.normalize_corpus(
									self.text_tokenizer.tokenize_corpus(self.text_extractor.get_texts(example))), 
								self.text_extractor.get_pearson(example), 
								self.text_extractor.get_category(example)) for example in examples[file_name]]
			partitions = self.text_extractor.get_text_partitions(examples[file_name], dictionary_partition=partitions)
		for category in partitions.keys():
			partitions_normalized[category] = self.text_normalizer.normalize_corpus(
				self.text_tokenizer.tokenize_corpus(partitions[category]))
			terms[category] = list(itertools.chain(*partitions_normalized[category]))
		processed_text += list(partitions_normalized.values())
		return examples, processed_text, partitions_normalized, terms, sts_dataset

	def words(self):
		return list(itertools.chain(*self.terms.values()))

	def sts_parts(self, category='all'):
		sentences, pearsons, = [], []
		for example in self.sts_dataset:
			if (category != 'all' and category == example[2]) or category == 'all':
				if example[0][0] and example[0][1]:
					sentences.append(example[0])
					pearsons.append(float(example[1]))
		return sentences, pearsons
