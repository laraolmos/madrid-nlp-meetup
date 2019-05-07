
# Madrid NLP Meetup Session: Challenges in Transfer Learning for NLP

This repository contains the material about "Challenges in Transfer Learning for NLP" event. In this Meetup we are going to experiment different sentences representation related with Transfer Learning in NLP models for the task of **Semantic Textual Similarity** (STS). 

In an simple example and tutorial, after pre-processing the official STS Benchmark from SemEval 2017, we obtain pairs of similar sentences (each phrase token list) and a similarity puntuation. Later we use different models to represent final sentence: pre-trained word embedding from FastText and Universal Sentence Encoder. Cosine function is used to see word and sentences similarity and Pearson coefficient to compare the similarities obtained with the gold standard.


## Dependencies to install

It is enough to work with Windows and Python 3 installed. With pip, install Jupyter and the following packages: nltk, codecs, numpy, matplotlib, seaborn, scipy, tensorflow and tensorflow_hub.


## Datasets to download

Dataset STSBenchmark: http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz
FastText English pre-trained word embeddings: https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip


## References and further reading

T. Mikolov, E. Grave, P. Bojanowski, C. Puhrsch, A. Joulin. Advances in Pre-Training Distributed Word Representations

Cer, D., Yang, Y., Kong, S. Y., Hua, N., Limtiaco, N., John, R. S., ... & Sung, Y. H. (2018). Universal sentence encoder. arXiv preprint arXiv:1803.11175. See in arXiv

Cer, D., Diab, M., Agirre, E., Lopez-Gazpio, I., & Specia, L. (2017). Semeval-2017 task 1: Semantic textual similarity-multilingual and cross-lingual focused evaluation. arXiv preprint arXiv:1708.00055. https://arxiv.org/pdf/1708.00055.pdf

SentEval: evaluation toolkit for sentence embeddings. https://github.com/facebookresearch/SentEval

Advances in Semantic Textual Similarity. https://ai.googleblog.com/2018/05/advances-in-semantic-textual-similarity.html
