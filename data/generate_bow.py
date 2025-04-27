import csv
import nltk
import pickle
import string
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


BATCH_SIZE = 500

for label in ('ai', 'human'):
    if label == 'ai':
        texts = ai_texts
    else:
        texts = human_texts

    i = 0
    for i in range(0, 1, BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]

        print("Generating encodings for {} text using bag of words".format(label))
        all_sentences = []
        all_embeddings = []
        for text in batch:
            sentences = sent_tokenize(text)
            all_sentences.append(sentences)

            proceesed_sentences = []
            for sentence in sentences:
                no_punc_sentence = sentence.translate(str.maketrans('', '', string.punctuation))

                filtered_text = []

                for word in word_tokenize(no_punc_sentence):
                    filtered_text.append(wnl.lemmatize(word, pos="v").lower())

                post_filtered_text = ' '.join(filtered_text)
                proceesed_sentences.append(post_filtered_text)
            vectorizer = CountVectorizer()
            embeddings = vectorizer.fit_transform(proceesed_sentences)
            all_embeddings.append(embeddings.toarray())
        write_to_file(all_sentences, all_embeddings, 'bow', i)

