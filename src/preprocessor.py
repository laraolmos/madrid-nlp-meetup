# -*- coding: utf-8 -*-
__author__ = 'Lara Olmos Camarena'


import re
import nltk.data
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords



# Segment text in phrases or words

class TextTokenizer():

	def sentences(self, input_text):
		english_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
		return english_tokenizer.tokenize(input_text)

	def words(self, input_text):
		return word_tokenize(input_text)

	def tokenize_corpus(self, corpus):
		return [self.words(text) for text in corpus]


# Process after tokenization

class TextNormalizer():

	def __init__(self):
		#self.stemmer = SnowballStemmer('spanish')
		self.stemmer = SnowballStemmer('english')

	def _remove_puntuation(self, word):
		regular_expr = re.compile('\r|\n|\t|\(|\)|\[|\]|:|\.|\,|\;|"|”|…|»|“|/|\'|\?|\¿|\!|\¡|`|\%|\.\.\.|-|—|=|–|―|@|#')
		word_processed = re.sub(regular_expr, '', word)
		#regex = re.compile('[^A-Za-z ]')
		#result = regex.sub('', word_processed)
		return word_processed

	def _filter_words(self, sentence):
		return [token for token in sentence if token not in [' ', ''] and len(token) < 10 ]

	# not used
	def _remove_numbers(self, word):
		num_expr = re.compile('[0-9]+|[0-9]*[,.][0-9]+')
		word_processed = re.sub(num_expr, '', word)
		return word_processed

	def _remove_stopwords(self, sentence):
		return [token for token in sentence if token not in stopwords.words('english')]

	def _stemming(self, token):
		return self.stemmer.stem(token)

	# normalization pipeline
	def normalize_token_list(self, token_list):
		processed_token_list = []
		for word in token_list:
			word_processed = word.lower().strip()
			word_processed = self._remove_puntuation(word_processed)
			word_processed = self._remove_numbers(word_processed)
			processed_token_list.append(word_processed)
		processed_token_list = self._filter_words(processed_token_list)
		return processed_token_list

	def normalize_corpus(self, corpus):
		return [self.normalize_token_list(token_list) for token_list in corpus]
