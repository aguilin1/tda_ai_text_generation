import csv
import pickle
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize

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

def write_to_file(sentences, embeddings, encoding_type, i = 1):
    with open("{}_encodings_{}-{}.pkl".format(encoding_type, label, i), "wb") as out:
        pickle.dump({'sentences': sentences, 'embeddings': embeddings},out)

START_AT = 0
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
