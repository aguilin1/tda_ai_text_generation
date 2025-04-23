import string
import nltk
import csv
import pickle
import itertools
from collections import deque
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

DATA_FILE = 'research-abstracts-labeled.csv'

human_texts = []
ai_texts = []
with open(DATA_FILE) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    next(csv_reader, None) # Skip header row
    for row in csv_reader:
      if row[1] == '0':
        human_texts.append(row[2])
      else:
        ai_texts.append(row[2])

nltk.download('punkt_tab')
nltk.download('wordnet')

wnl = WordNetLemmatizer()

def write_to_file(sentences, embeddings, encoding_type, i = 1):
    with open("{}_encodings_{}-{}.pkl".format(encoding_type, label, i), "wb") as out:
        pickle.dump({'sentences': sentences, 'embeddings': embeddings},out)


START_AT = 8700
BATCH_SIZE = 100

label = 'human'
if label == 'ai':
    texts = ai_texts
else:
    texts = human_texts

i = 0
for i in range(START_AT, len(texts), BATCH_SIZE):
    batch = texts[i:i+BATCH_SIZE]
    sentence_batch = []
    embeddings_batch = []
    i = i + 1
    print("Generating encodings for {} text using BERT (batch {})".format(label, i))
    j = 0
    for text in batch:
        j += 1
        print("Processing {}/100 in batch".format(j))
        bert_model = SentenceTransformer("all-MiniLM-L6-v2")
        sentences = sent_tokenize(text)
        sentence_batch.append(sentences)
        tokens = sentences
        embeddings = bert_model.encode(sentences)
        embeddings_batch.append(embeddings)
    write_to_file(sentence_batch, embeddings_batch, 'sentence-bert', i)

# test = 'bow_encodings_ai.pkl'
# for test in ('sentence-bert_encodings_human-2.pkl', 'bow_encodings_ai-1.pkl'):
#     with open(test, "rb") as fIn:
#         cache_data = pickle.load(fIn)
#         corpus_sentences = cache_data['sentences']
#         print(len(corpus_sentences))
#         corpus_embeddings = cache_data['embeddings']

#     import numpy as np
#     from sklearn.metrics.pairwise import cosine_similarity
#     import itertools

#     # Label each sentence starting at 0, 1, 2, ...
#     # Track distances as ((sentence 1, sentence 2), distance)
#     distances = []
#     for i in range(2):
#         sentences = corpus_sentences[i]
#         embeddings = corpus_embeddings[i]
#         for pair1_i, pair2_i in itertools.combinations(range(len(sentences)), 2):
#             cos_sim = cosine_similarity(np.reshape(embeddings[pair1_i], (1, -1)), np.reshape(embeddings[pair2_i], (1, -1)))[0][0]
#             dist = 2 * np.arccos(cos_sim) / np.pi
#             distances.append(((pair1_i, pair2_i), dist))

    # for distance in distances:
    #     print("Distance {} from [{}] to [{}]".format(distance[1], sentences[distance[0][0]], sentences[distance[0][1]]))
