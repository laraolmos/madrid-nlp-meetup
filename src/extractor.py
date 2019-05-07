# -*- coding: utf-8 -*-
__author__ = 'Lara Olmos Camarena'


import codecs


class STSExtractor():

	''' Clean row example line with tab separator
	Header: category dataset partition numExample PearsonCoeficient FirstText SecondText
	Example: main-captions	MSRvid	2012test	0000	5.000	A man with a hard hat is dancing.	A man wearing a hard hat is dancing.
	'''
	def extract_example(self, example_row):
		splitted = example_row.split('\t')
		if splitted and len(splitted) > 6:
			return splitted

	# all content HTML file in one string
	def extract_file_examples(self, file_route):
		with codecs.open(file_route, 'r', encoding='utf-8')  as content_file:
			example_rows = content_file.read().split('\n')
			dataset = [self.extract_example(example_row) for example_row in example_rows if self.extract_example(example_row) != None]
		return dataset

	def get_category(self, example):
		if example[0] == 'main-forum':
			return 'main-forums'
		return example[0]

	def get_dataset(self, example):
		return example[1]

	def get_partition(self, example):
		return example[2]

	def get_pearson(self, example):
		return example[4]

	def get_texts(self, example):
		return [example[5], example[6]]

	def get_text_partitions(self, dataset, dictionary_partition={}):
		for example in dataset:
			category = self.get_category(example)
			if category == 'main-forum':
				category = 'main-forums'
			if category in dictionary_partition:
				texts = self.get_texts(example)
				dictionary_partition[category].append(texts[0])
				dictionary_partition[category].append(texts[1])
			else:
				dictionary_partition[category] = []
		return dictionary_partition
