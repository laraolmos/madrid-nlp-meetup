# -*- coding: utf-8 -*-
__author__ = 'Lara Olmos Camarena'


import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns



class UnivSentEncoder():

	def __init__(self, model_route):
		self.model_route = model_route
		self.embed_model = None

	def encode_sentences(self, sentence_list):
		sentences_placeholder = tf.placeholder(tf.string, shape=(None))
		sentence_encodings = self.embed_model(sentences_placeholder)
		with tf.Session() as session:
			session.run([tf.global_variables_initializer(), tf.tables_initializer()])
			sentence_embedding = session.run(sentence_encodings, feed_dict={sentences_placeholder: sentence_list})
		return sentence_embedding

	def show_info(self, np_embedding, sentence_list):
		for i, message_embedding in enumerate(np.array(np_embedding).tolist()):
			print("Sentence: {}".format(sentence_list[i]))
			print("Embedding size: {}".format(len(message_embedding)))
			message_embedding_snippet = ", ".join((str(x) for x in message_embedding[:3]))
			print("Embedding: [{}, ...]\n".format(message_embedding_snippet))

	def plot_similarity(self, labels, features, model_ref):
		corr = np.inner(features, features)
		sns.set(font_scale=0.8)
		plt.figure(figsize=(12,6))
		sns_heatmap = sns.heatmap(corr, xticklabels=labels, yticklabels=labels, vmin=0, vmax=1, cmap="YlOrRd")
		sns_heatmap.set_xticklabels(labels, rotation=90)
		sns_heatmap.set_title("STS " + model_ref)
		plt.tight_layout()
		plt.savefig("STS " + model_ref + ".png")
		plt.clf()

	# TODO
	def export_sentence_encodings(self):
		pass



#########################################################
# UNIVERSAL SENTENCE ENCODER
#########################################################

class UnivSentEncTransformer(UnivSentEncoder):

	def __init__(self, model_route="https://tfhub.dev/google/universal-sentence-encoder-large/3"):
		super(UnivSentEncTransformer, self).__init__(model_route)
		self.embed_model = hub.Module(self.model_route)


class UnivSentEncDAN(UnivSentEncoder):

	def __init__(self, model_route="https://tfhub.dev/google/universal-sentence-encoder/2"):
		super(UnivSentEncDAN, self).__init__(model_route)
		self.embed_model = hub.Module(self.model_route)


class UnivSentEncSpanish(UnivSentEncoder):

	def __init__(self, model_route="https://tfhub.dev/google/universal-sentence-encoder-xling/en-es/1"):
		super(UnivSentEncSpanish, self).__init__(model_route)
		self.embed_model = hub.Module(self.model_route)



#########################################################
# TEST
#########################################################

def test_execution(model_name, sentence_list):
	import time
	start_time = time.time()
	if model_name == 'univ-sent-enc-transformer':
		sentence_encoder = UnivSentEncTransformer()
	if model_name == 'univ-sent-enc-dan':
		sentence_encoder = UnivSentEncDAN()
	if model_name == 'univ-sent-enc-spa':
		sentence_encoder = UnivSentEncSpanish()
	encoding_result = sentence_encoder.encode_sentences(sentence_list)
	elapsed_time = time.time() - start_time
	print('Execution time ' + model_name + ': ' + str(elapsed_time))
	return sentence_encoder, encoding_result


def test_model(model_name, model_ref, sentence_list):
	sentence_encoder, model_result = test_execution(model_name, sentence_list)
	#sentence_encoder.show_info(model_result, sentence_list)
	sentence_encoder.plot_similarity(sentence_list, model_result, model_ref)


if __name__ == '__main__':

	english_messages = [
	    # Smartphones
	   "I like my phone",
	   "My phone is not good.",
	   "Your cellphone looks great.",
	   # Weather
	   "Will it snow tomorrow?",
	   "Recently a lot of hurricanes have hit the US",
	   "Global warming is real",
	   # Food and health
	   "An apple a day, keeps the doctors away",
	   "Eating strawberries is healthy",
	   "Is paleo better than keto?",
	   # Asking about age
	   "How old are you?",
	   "what is your age?"
	]

	spanish_messages = [
	    # Smartphones
	    "Me gusta mi teléfono",
	    "Mi teléfono no es bueno",
	    "Tu móvil es genial",
	    # Weather
	    "¿Nevará mañana?",
	    "Muchos huracanes han dañado US recientemente",
	    "El calentamiento global es real",
	    # Food and health
	    "Una manzana al día mantiene a los doctores lejos",
	    "Comer fresas es sano",
	    "¿Es mejor paleo o keto?",
	    # Asking about age
	    "¿Cuántos años tienes?",
	    "¿Qué edad tienes?"
	]

	test_model('univ-sent-enc-transformer', 'Univ Sent Enc Trans ENG', english_messages)
	test_model('univ-sent-enc-dan', 'Univ Sent Enc DAN ENG', english_messages)

	# WIP, TODO Not working!
	#test_model('univ-sent-enc-spa', 'Univ Sent Enc SPA', spanish_messages)