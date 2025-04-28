import csv
import pdb
import pandas as pd
import numpy as np
from transformers import pipeline
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
import evaluate
import sys
import os
# Model cache
os.environ['HF_HOME'] = '/data/900G/Project/data/cache'

data = []
df = pd.read_csv('./research-abstracts-labeled.csv', nrows=5000)
label = df["label"]
text = df["text"]
wc = df["word_count"]
df["all_text"] = df.apply(
        lambda r: " ".join(
            [
                str(r["label"]) if pd.notnull(r["label"]) else "",
                str(r["text"]) if pd.notnull(r["text"]) else ""
            ]
        ), axis = 1
)
label_encoder = LabelEncoder()

print (label.shape, text.shape, wc.shape)

i =0
j = 0
for k in range(5000):
    if label[k] == 0:
        i = i+ 1
    else:
        j = j + 1

print ("Human Sentences : ", i, "AI Sentences : ", j)
def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
"""
with open('./research-abstracts-labeled.csv', newline='\n') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
    header = next(spamreader)
    #print(header)
    column_names = ['label', "text", "word_count"]
    column_indices = [header.index(name) for name in column_names]
    print(column_indices)
    i = 0
    for row in spamreader:
        i = i+1
        if i > 5000:
            break
        print("Processing line : ", i)
        if len(row) < 3:
            continue
        else :
            selected_values = [row[i] for i in column_indices]
            row_data = dict(zip(column_names, selected_values))
            data.append(row_data)
            print(row_data)
"""
#############################################################################
# https://medium.com/freelancers-hub/best-ai-detectors-2025-35a58eac86c5
# Top 2 failed : AI Detector, Phrasly.AI, Originality.AI had its own sample.
#                https://undetectable.ai/?_by=n8q92 had its own samples.
# Systems with their own  samples will not let me enter text from GPT-4.
# Radar - https://radar-app.vizhub.ai/ - 64% AI.
# GLTR - Many Green tokens, most likely AI
#############################################################################
print("Read 5000 Sentences! Done")

human_1 = "Coupling losses were studied in composite tapes containing superconducting material in the form of two separate stacks of densely packed filaments embedded in a metallic matrix of Ag or Ag alloy. This kind of sample geometry is quite favorable for studying the coupling currents and in particular the role of superconducting bridges between filaments. By using a.c. susceptibility technique, the electromagnetic losses as function of a.c. magnetic field amplitude and frequency were measured at the temperature T = 77 K for two tapes with different matrix composition. The length of samples was varied by subsequent cutting in order to investigate its influence on the dynamics of magnetic flux penetration. The geometrical factor $\chi_0$ which takes into account the demagnetizing effects was established from a.c. susceptibility data at low amplitudes. Losses vs frequency dependencies have been found to agree nicely with the theoretical model developed for round multifilamentary wires. Applying this model, the effective resistivity of the matrix was determined for each tape, by using only measured quantities. For the tape with pure silver matrix its value was found to be larger than what predicted by the theory for given metal resistivity and filamentary architecture. On the contrary, in the sample with a Ag/Mg alloy matrix, an effective resistivity much lower than expected was determined. We explain these discrepancies by taking into account the properties of the electrical contact of the interface between the superconducting filaments and the normal matrix. In the case of soft matrix of pure Ag, this is of poor quality, while the properties of alloy matrix seem to provoke an extensive creation of intergrowths which can be actually observed in this kind of samples."

machine_1 = "In this study, we investigate the coupling loss on bi-columnar BSCCO/Ag tapes using a.c. susceptibility measurements. The bi-columnar structure of BSCCO tapes is known to offer several advantages over traditional tape configurations, including increased tolerance to magnetic field disturbances. However, the effects of the Bi-2212/Ag interface on the coupling between the superconducting filaments of the BSCCO tape is not well understood. Our experiments show that the coupling loss is dominated by the Bi-2212/Ag interface and varies significantly with the orientation and magnitude of the applied a.c. magnetic field. Specifically, coupling loss is found to be lower for in-plane magnetic fields and higher for out-of-plane magnetic fields. We also observe that the annealing of the tapes significantly affects the coupling loss, as annealed tapes exhibit lower loss values than unannealed tapes. Furthermore, we find that the coupling loss is sensitive to the orientation of the Ag matrix, as demonstrated by measurements on tapes with both transverse and longitudinal matrix orientation. Finally, we use numerical simulations to confirm the validity of our experimental results. Overall, this study provides important insights into the coupling loss mechanisms in bi-columnar BSCCO/Ag tapes, which are highly relevant for the development of practical applications of high-temperature superconductors."

#human_2 = "Let $\mathsf M_{\mathsf S}$ denote the strong maximal operator on $\mathbb R^n$ and let $w$ be a non-negative, locally integrable function. For $\alpha\in(0,1)$ we define the weighted sharp Tauberian constant $\mathsf C_{\mathsf S}$ associated with $\mathsf M_{\mathsf S}$ by $$ \mathsf C_{\mathsf S} (\alpha):= \sup_{\substack {E\subset \mathbb R^n \\ 0<w(E)<+\infty}}\frac{1}{w(E)}w(\{x\in\mathbb R^n:\, \mathsf M_{\mathsf S}(\mathbf{1}_E)(x)>\alpha\}). $$ We show that $\lim_{\alpha\to 1^-} \mathsf C_{\mathsf S} (\alpha)=1$ if and only if $w\in A_\infty ^*$, that is if and only if $w$ is a strong Muckenhoupt weight. This is quantified by the estimate $\mathsf C_{\mathsf S}(\alpha)-1\lesssim_{n} (1-\alpha)^{(cn [w]_{A_\infty ^*})^{-1}}$ as $\alpha\to 1^-$, where $c>0$ is a numerical constant; this estimate is sharp in the sense that the exponent $1/(cn[w]_{A_\infty ^*})$ can not be improved in terms of $[w]_{A_\infty ^*}$. As corollaries, we obtain a sharp reverse H\"\"older inequality for strong Muckenhoupt weights in $\mathbb R^n$ as well as a quantitative imbedding of $A_\infty^*$ into $A_{p}^*$. We also consider the strong maximal operator on $\mathbb R^n$ associated with the weight $w$ and denoted by $\mathsf M_{\mathsf S} ^w$. In this case the corresponding sharp Tauberian constant $\mathsf C_{\mathsf S} ^w$ is defined by $$ \mathsf C_{\mathsf S} ^w \alpha) := \sup_{\substack {E\subset \mathbb R^n \\ 0<w(E)<+\infty}}\frac{1}{w(E)}w(\{x\in\mathbb R^n:\, \mathsf M_{\mathsf S} ^w (\mathbf{1}_E)(x)>\alpha\}).$$ We show that there exists some constant $c_{w,n}>0$ depending only on $w$ and the dimension $n$ such that $\mathsf C_{\mathsf S} ^w (\alpha)-1 \lesssim_{w,n} (1-\alpha)^{c_{w,n}}$ as $\alpha\to 1^-$ whenever $w\in A_\infty ^*$ is a strong Muckenhoupt weight."

human_2 = "Let $\mathsf M_{\mathsf S}$ denote the strong maximal operator on $\mathbb R^n$ and let $w$ be a non-negative, locally integrable function. For $\alpha\in(0,1)$ we define the weighted sharp Tauberian constant $\mathsf C_{\mathsf S}$ associated with $\mathsf M_{\mathsf S}$ by $$ \mathsf C_{\mathsf S} (\alpha):= \sup_{\substack {E\subset \mathbb R^n \\ 0<w(E)<+\infty}}\frac{1}{w(E)}w(\{x\in\mathbb R^n:\, \mathsf M_{\mathsf S}(\mathbf{1}_E)(x)>\alpha\}). $$ We show that $\lim_{\alpha\to 1^-} \mathsf C_{\mathsf S} (\alpha)=1$ if and only if $w\in A_\infty ^*$, that is if and only if $w$ is a strong Muckenhoupt weight. This is quantified by the estimate $\mathsf C_{\mathsf S}(\alpha)-1\lesssim_{n} (1-\alpha)^{(cn [w]_{A_\infty ^*})^{-1}}$ as $\alpha\to 1^-$, where $c>0$ is a numerical constant; this estimate is sharp in the sense that the exponent $1/(cn[w]_{A_\infty ^*})$ can not be improved in terms of $[w]_{A_\infty ^*}$. As corollaries, we obtain a sharp reverse H\"\"older inequality for strong Muckenhoupt weights in $\mathbb R^n$ as well as a quantitative imbedding of $A_\infty^*$ into $A_{p}^*$. We also consider the strong maximal operator on $\mathbb R^n$ associated with the weight $w$ and denoted by $\mathsf M_{\mathsf S} ^w$. In this case the corresponding sharp Tauberian constant $\mathsf C_{\mathsf S} ^w$ is defined by $$ \mathsf C_{\mathsf S} ^w \alpha) := \sup_{\substack {E\subset \mathbb R^n \\ 0<w(E)<+\infty}}\frac{1}{w(E)}w(\{x\in\mathbb R^n:\, \mathsf M_{\mathsf S} ^w (\mathbf{1}_E)(x)>\alpha\}).$$ We show that there exists some constant $c_{w,n}>0$ depending only on $w$ and the dimension $n$ such that $\mathsf C_{\mathsf S} ^w (\alpha)-1 \lesssim_{w,n} (1-\alpha)^{c_{w,n}}$ as $\alpha\to 1^-$ whenever $w\in A_\infty ^*$ is a strong Muckenhoupt weight."

machine_2 = "In this paper, we investigate Weighted Solyanik Estimates for the Strong Maximal Function. The purpose of this study is to determine the optimal range of weights for which the Solyanik estimates hold true for the strong maximal function in both dyadic and non-dyadic contexts. Our work is motivated by recent developments in the area of harmonic analysis and Fourier analysis, which have demonstrated the importance of the strong maximal function in many areas of mathematics. We begin our investigation by defining the strong maximal function and introducing the weighted Solyanik estimates. We then analyze the properties of the strong maximal function and its relationship to weighted estimates. Our findings demonstrate that the range of suitable weights for the Solyanik estimates is significantly larger in the dyadic context than in the non-dyadic context. Moreover, we establish new estimates for the strong maximal function in the dyadic context, which improve upon existing results. Our study contributes to the development of more precise and accurate techniques for analyzing the strong maximal function, and may have potential applications in other areas of mathematics as well. In conclusion, our investigation of Weighted Solyanik Estimates for the Strong Maximal Function provides new insights into this important topic in harmonic analysis, and we expect our results to be of interest to researchers in this field and related areas of study."

gpt_4 = "Ukraine is a country in Eastern Europe, known for its rich cultural heritage, resilient people, and strategic geographical location. Covering more than 600,000 square kilometers, it is the second-largest country in Europe after Russia. Kyiv, its capital, stands as one of the oldest cities in Eastern Europe, with a history dating back over 1,400 years. Ukraine's landscape varies from fertile plains and steppes to the majestic Carpathian Mountains in the west and the Black Sea coastline to the south. It has long been recognized as a global breadbasket, thanks to its vast agricultural resources. Ukrainian culture reflects a blend of Eastern European traditions, with strong influences from neighboring Poland, Russia, and Turkey. Folk music, vibrant dances like the Hopak, and intricate embroidery are central to its national identity. Since 2014, Ukraine has been at the forefront of international attention due to conflicts stemming from Russia’s annexation of Crimea and ongoing tensions in the Donbas region. Despite facing war and economic challenges, Ukrainians have shown remarkable unity and determination in defending their sovereignty and pushing toward integration with Western institutions like the European Union and NATO. Today, Ukraine symbolizes the struggle for democracy, independence, and self-determination in the modern world."

deepseek = "Ukraine, located in Eastern Europe, is the second-largest country on the continent by area, bordered by Russia, Belarus, Poland, Slovakia, Hungary, Romania, Moldova, and the Black Sea. With a rich cultural heritage, Ukraine has been a crossroads of civilizations for centuries, influenced by Slavic, Byzantine, and Western European traditions. Kyiv, its capital and largest city, is one of the oldest cities in Eastern Europe, known for its golden-domed churches like Saint Sophia Cathedral and Kyiv Pechersk Lavra. Ukraine gained independence from the Soviet Union in 1991 and has since pursued a path toward democracy and European integration. However, its sovereignty has been challenged by Russian aggression, including the illegal annexation of Crimea in 2014 and the ongoing full-scale war since February 2022. Despite immense destruction and humanitarian crises, Ukraine has shown remarkable resilience, with its people and military fiercely defending their freedom. The country is known for its fertile lands, often called the \"breadbasket of Europe,\" and its contributions to literature, music, and arts, with figures like Taras Shevchenko and Lesya Ukrainka playing key roles in national identity. Today, Ukraine continues to fight for its survival and future, seeking support from Western allies while striving for reforms and EU and NATO membership."

grok = "Ukraine, a vibrant nation in Eastern Europe, boasts a rich tapestry of history, culture, and resilience. Spanning over 600,000 square kilometers, it is the largest country entirely within Europe, bordered by Poland, Russia, and the Black Sea. Kyiv, its capital, is a hub of ancient architecture, with landmarks like the golden-domed St. Sophia’s Cathedral and the sprawling Pechersk Lavra monastery. Ukraine’s cultural heritage shines through its colorful traditions, including intricate embroidery, lively folk dances like the hopak, and a cuisine featuring borscht and varenyky. Despite facing significant challenges, including the ongoing conflict sparked by Russia’s 2014 annexation of Crimea and the 2022 invasion, Ukraine’s people have shown remarkable strength and unity. The nation’s fertile black soil, known as “chernozem,” makes it a global breadbasket, producing vast quantities of wheat and sunflower oil. Ukraine’s push for sovereignty and democratic values has inspired millions, with its blue-and-yellow flag symbolizing hope and freedom. From the Carpathian Mountains to the Black Sea coast, Ukraine’s diverse landscapes and indomitable spirit continue to captivate and inspire, reflecting a nation determined to shape its own future."

#sentences = ["I ate dinner", "We had a three course meal", "Colin came to have dinner with us", "He loves tacos and fish", "In the end we all felt like we ate too much", "We all agreed that it was a magnificent evening"]
#sentences = [human_1, human_2, machine_1, machine_2, gpt_4]
#sentences = [human_1, human_2, machine_1, machine_2, deepseek]
sentences = [human_1, human_2, machine_1, machine_2, grok]

from sentence_transformers import SentenceTransformer, InputExample, LoggingHandler, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from datasets import DatasetDict, Dataset, load_dataset
from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding
from transformers import AutoTokenizer, Trainer, TrainingArguments
#dataset_dict = load_dataset('./research-abstracts-labeled.csv')
#dataset_dict = load_dataset('./data7000.csv')
#dataset_dict = load_dataset('./data7000.csv')

# Login using e.g. `huggingface-cli login` to access this dataset
dataset_dict = load_dataset("NicolaiSivesind/human-vs-machine", "research_abstracts_labeled")
print(dataset_dict['train'][0])

def prepare_features(sentence):
    #print("SEN", sentence) 
    features = {'text': sentence['text'], 'label': sentence['label']}
    # removes title, word_count etc..
    #print("FTR", features)
    return features
dataset_dict = dataset_dict.map(prepare_features)
#print("AFTER Prep", dataset_dict['train'][0])
# debugging dataload
#sys.exit(0)

# Pretrained SBERT
# tokenizer = AutoTokenizer.from_pretrained('./final/')
# Classification head
import torch
import torch.nn as nn
class ClassificationHead(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(ClassificationHead, self).__init__()
        self.linear = nn.Linear(embedding_dim, num_classes)

    def forward(self, features):
        x = features['sentence_embedding']
        x = self.linear(x)
        return x

# Number of classes (Machine, Human)
num_classes = 2
print('Class Number : ', num_classes)
#model_name='sentence-transformers/all-MiniLM-L6-v2'
#model_name='sentence-transformers/all-MiniLM-L6-v2'
model_name = 'google-bert/bert-base-uncased'
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')

# Going with AutoModalForSequenceClassification instead of BertForSequenceClassification as 
# AutoModal has Classicationhead, that can be trained layer by layey, while freezing lower level layers
# This brings down training costs.

from transformers import BertForSequenceClassification
model_with_head = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Bye SBERT Apr26, 2025
#model = SentenceTransformer('all-MiniLM-L6-v2')
#classification_head = ClassificationHead(model.get_sentence_embedding_dimension(), num_classes)
"""
# Combine SentenceTransformer model and classification head
class SentenceTransformerWithHead(nn.Module):
    def __init__(self, transformer, head, device, card, tokenize, tokenizer):
        super(SentenceTransformerWithHead, self).__init__()
        self.transformer = transformer
        self.head = head
        self.device = device
        self.model_card_data = card
        self.tokenize = tokenize
        self.tokenizer = tokenizer

    def forward(self, input):
        features = self.transformer(input)
        logits = self.head(features)
        return logits
"""
id2label = {0: "human", 1: "AI"}
label2id = {"human": 0, "AI": 1}
#model_with_head = AutoModelForSequenceClassification.from_pretrained(model_name, num_classes, id2label=id2label, label2id=label2id,)
#model_with_head = AutoModelForSequenceClassification.from_pretrained(model_name, num_classes)

#Following are for SBERT
#model_with_head = AutoModelForSequenceClassification.from_pretrained(model_name, num_classes)
#model_with_head = SentenceTransformerWithHead(model, classification_head, model.device, model.model_card_data, model.tokenize, tokenizer)
#Train

from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
""" SBERT MOdel
train_sentences = df['all_text']
train_labels = df['label']
num_epochs = 10
batch_size=10
learning_rate = 2e-5

train_examples = [InputExample(texts=[s], label=l) for s, l in zip(train_sentences, train_labels)]

def collate_fn(batch):
    texts = [example.text[0] for example in batch]
    labels = torch.tensor([example.label for example in batch])
    return texts, labels

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size, collate_fn=collate_fn)
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model_with_head.parameters(), lr=learning_rate)
total_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps = total_steps)
"""
# Freeze lower level layers while training classification head and last 2 pooling layers
for name, param in model_with_head.base_model.named_parameters():
    param.required_grad = False

for name, param in model_with_head.base_model.named_parameters():
    if "pooler" in name:
        param.required_grad = True
def preprocess_function(sentences):
    return tokenizer(sentences['text'], truncation=True)

tokenized_data = dataset_dict.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

print(tokenized_data["train"].features['input_ids'])

metric = evaluate.load("accuracy")
auc_score = evaluate.load("roc_auc")

def compute_metrics(eval_pred):
    pred, labels = eval_pred
    prob = np.exp(pred) / np.exp(pred).sum(-1, keepdims=True)
    positive_class = prob[:,1]
    #perplexity = evaluate.load("/data/900G/Project/data/my_perplexity", module_type="metric")
    #results = metric.compute(
    #        model_id=(model, tokenizer),
    #        predictions=predictions
    #        )
    #auc = np.round(auc_score.compute(prediction_score=positive_class, reference=labels)['roc_auc'], 3)
    pred_classes = np.argmax(pred, axis =1)
    acc = np.round(metric.compute(predictions=pred_classes, references=labels)['accuracy'],3)
    #return {"Accuracy": acc, "AUC":auc}
    return {"Accuracy": acc }

learning_rate = 2e-5
num_epochs = 10
batch_size=10
#training_args = SentenceTransformerTrainingArguments(
training_args = TrainingArguments(
    output_dir = "AI_classifier",
    learning_rate = learning_rate,
    per_device_train_batch_size = batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    logging_strategy="epoch",
    eval_strategy="epoch",
    save_strategy="epoch",
    #remove_unused_columns=False, - needed for SBERT
    load_best_model_at_end=True,)

"""
# Needed for SBERT
model_with_head = model_with_head.to(model.device)
tokenized_data = tokenized_data.remove_columns("title")
tokenized_data = tokenized_data.remove_columns("text")
tokenized_data = tokenized_data.remove_columns("word_count")
tokenized_data = tokenized_data.remove_columns("label")
trainer = SentenceTransformerTrainer(
"""

""" BERT Training
trainer = Trainer(
    model=model_with_head,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    )

# Checks
print(trainer.train_dataset[0])

trainer.train()
predictions = trainer.predict(tokenized_data["validation"])
logits = predictions.predictions
labels = predictions.label_ids
metrics = compute_metrics((logits, labels))
print(metrics)

model_with_head.save_pretrained('./my_bert_model')
tokenizer.save_pretrained('./my_bert_model')
"""

# AFTER Training see predictions:

from transformers import BertForSequenceClassification
tokenizer = BertTokenizer.from_pretrained('./my_bert_model')
model_with_head = BertForSequenceClassification.from_pretrained('./my_bert_model', num_labels=2)

model_with_head.eval()

def predict(text):
    marked_text = "[CLS] " + text + " [SEP]"
    inputs = tokenizer(marked_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    print(inputs)
    with torch.no_grad():
        outputs = model_with_head(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
    return "AI " if predictions.item()==1 else "Human"

marked_text_1 = "[CLS] " + human_1 + " [SEP]"
marked_text_2 = "[CLS] " + human_2 + " [SEP]"
marked_text_3 = "[CLS] " + machine_1 + " [SEP]"
marked_text_4 = "[CLS] " + machine_2 + " [SEP]"
#marked_text_5 = "[CLS] " + gpt_4 + " [SEP]"
#marked_text_5 = "[CLS] " + deepseek + " [SEP]"
marked_text_5 = "[CLS] " + grok + " [SEP]"
input_tokens_1 = tokenizer.encode_plus(marked_text_1, add_special_tokens=True, 
                                       return_tensors="pt", padding=True, truncation=True).input_ids
input_tokens_2 = tokenizer.encode_plus(marked_text_2, add_special_tokens=True, 
                                       return_tensors="pt", padding=True, truncation=True).input_ids
input_tokens_3 = tokenizer.encode_plus(marked_text_3, add_special_tokens=True, 
                                       return_tensors="pt", padding=True, truncation=True).input_ids
input_tokens_4 = tokenizer.encode_plus(marked_text_4, add_special_tokens=True, 
                                       return_tensors="pt", padding=True, truncation=True).input_ids
input_tokens_5 = tokenizer.encode_plus(marked_text_5, add_special_tokens=True, 
                                       return_tensors="pt", padding=True, truncation=True).input_ids
print(input_tokens_1)
outputs_1 = model_with_head(input_tokens_1, output_hidden_states=True)
outputs_2 = model_with_head(input_tokens_2, output_hidden_states=True)
outputs_3 = model_with_head(input_tokens_3, output_hidden_states=True)
outputs_4 = model_with_head(input_tokens_4, output_hidden_states=True)
outputs_5 = model_with_head(input_tokens_5, output_hidden_states=True)
print(len(outputs_1))
# Shape is 1 * Number of tokens * 768
# We pick CLS embedding
print(outputs_1.hidden_states[-1].shape)
query_vec_CLS_1 = outputs_1.hidden_states[-1][0][0]
query_vec_CLS_2 = outputs_2.hidden_states[-1][0][0]
query_vec_CLS_3 = outputs_3.hidden_states[-1][0][0]
query_vec_CLS_4 = outputs_4.hidden_states[-1][0][0]
query_vec_CLS_5 = outputs_5.hidden_states[-1][0][0]
print(query_vec_CLS_1.shape, outputs_1.hidden_states[-1][0].shape)
cls_embeddings = [query_vec_CLS_1, query_vec_CLS_2, query_vec_CLS_3, query_vec_CLS_4, query_vec_CLS_5]
ind = 0
# One can see cosine similarities between human vs AI text
with torch.no_grad():
    for sent in sentences:
        print(predict(sent))
        sim = cosine(query_vec_CLS_1.numpy(), cls_embeddings[ind].numpy())
        ind = ind + 1
        print("Sentence = ", sent, "; similarity = ", sim, "Vector Shape = ", query_vec_CLS_1.shape)

DATA_COUNT=200
LAYER = 11
input_emb=[]
text_df = df["text"]
for i in range(25, 25+DATA_COUNT):
    emb = tokenizer.encode_plus(
            text_df[i], 
            add_special_tokens=True, 
            return_tensors='pt',
            padding=True,
            truncation=True)['input_ids']
    input_emb.append(emb)

#print(input_emb, input_emb[0])

hs = []
with torch.no_grad():
    for i in range(DATA_COUNT):
        outputs = model_with_head(input_emb[i], output_hidden_states=True)
        #hidden_states = outputs.last_hidden_state
        #last hidden state contains the embedding for all layers.
        # only saving 768 byte vector for start of paragraph "[CLS] " token's embeddings.
        # First token embeddings
        query_vec = outputs.hidden_states[LAYER][0][0]
        print(len(query_vec), query_vec.shape)
        hs.append(query_vec)

print(hs[0].shape)
# We will use CLS embedding of first token of paragraphs as classsfication.
#first_token_embedding = hs[0][0][0]
first_paragraph_embedding = hs[0]
print(first_paragraph_embedding)

# We can also normalize embeddings for entire paragraph and use that for classification.
paragraph_emb = []

# only write 500 as file is getting larger
for k in range(DATA_COUNT):
    # just store last layer's embeddings https://huggingface.co/docs/transformers/model_doc/bert#bertmodel
    paragraph_embedding = hs[k]
    #paragraph_embedding = torch.mean(hs[k][0], dim=0)
    print(paragraph_embedding.shape)
    paragraph_emb.append(paragraph_embedding)

# Size debug: BERT CLS encodings are being stored by pickle as 7K*768 byte vector for some reason.
#for k in paragraph_emb:
#    print(k.shape)
#    print(k)

print(len(paragraph_emb))
import pickle
FILENAME = "my_dataset_bert_cls_" + str(DATA_COUNT) + "_" + str(LAYER) + ".pickle"
with open(FILENAME, "wb") as pkl:
    pickle.dump(paragraph_emb, pkl)

print("Done with all ", DATA_COUNT, "For LAYER ", LAYER, " paragraphs.", len(paragraph_emb), len(paragraph_emb[0]))

"""
# SBERT with ClassificationHead training, after training modal's pooling layer though everything
# is similar to everything else thus assinging similarity as 0.85 to human1, machine1, machine1 and 1 to direct match
# human2. I compared human2 with [human1, huma2, machine1, machine2], to see whether AI text category is getting recognized.

loss_list=[]
for epoch in range(num_epochs):
    model_with_head.train()
    for step, (texts, labels) in enumerate(train_dataloader):
        labels = labels.to(model.device)
        optimizer.zero_grad()

        # Encode
        inputs = model.tokenize(texts)
        input_ids = inputs['input_ids'].to(model.device)
        input_attention_mask = inputs['attention_mask'].to(model.device)
        inputs_final = {'input_ids' : input_ids, 'attention_mask':input_attention_mask}

        # model with head
        model_with_head = model_with_head.to(model.device)
        logits = model_with_head(inputs_final)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 100 == 0:
            print(f"Epoch {epoch}, Step {step}, Loass : {loss.item()}")

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
        model_save_path = f'./imtermediate-output/epoch-{epoch}'
        model.save(model_save_path)
        loss_list.append(loss.item())

    #save final
    model_final_save_path='final-1'
    model.save(model_final_save_path)

from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score

#sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
#sbert_model = SentenceTransformer('all-mpnet-base-v2')
#sbert_model = SentenceTransformer('all-distilroberta-v1')
sbert_model = SentenceTransformer('final')
data['text_embeddings'] = df['all_text'].apply(lambda x: model.encode(str(x)))
text_embeddings = pd.DataFrame(data['text_embeddings'].tolist(), index=data.index, dtype=float)
X_0 = df['label']
X = pd.concat([X_0, text_embeddings], axis=1)
#Convert numeric columns to float
label_encoder = LabelEncoder()
df['category'] = label_encoder.fit_transform(df['label'])
y = df['category']
X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.2, random_state=42)

unique_labels = sorted(set(y_train) | set(y_test))
label_mapping = {label: idx for idx, label in enumerate(unique_labels)}

y_train = y_train.map(label_mapping)
y_test = y_test.map(label_mapping)

dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)
param = {
        'max_depth' : 6,
        'eta' : 0.3,
        'objective': 'multi:softmax',
        'num_class': len(label_mapping),
        'eval_metric': 'mlogloss'
        }

num_round = 100
bst = xgb.train(param, dtrain, num_round)
y_pred = sbert_model.predict(dtest)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy : {accuracy:.2f}')

from transformers import AutoTokenizer
sbert_model = SentenceTransformer('final')
tokenizer = AutoTokenizer.from_pretrained('./final/')

inputs = sbert_model.tokenize(human_1)
input_ids = inputs['input_ids'].to(model.device)
input_attention_mask = inputs['attention_mask'].to(model.device)
inputs_final = {'input_ids' : input_ids, 'attention_mask':input_attention_mask}

# model with head
model_with_head = model_with_head.to(model.device)
#criterion = nn.CrossEntropyLoss()
logits = model_with_head.predict(inputs_final)
preds = torch.max(logits, dim=1)
print(preds)
#print(F.sigmoid(logits).squeeze(1)[0])
#print(F.sigmoid(logits).squeeze(1).sum())
#print(F.sigmoid(logits).squeeze(1).mean())
#loss = criterion(logits, labels)
tokens_human = tokenizer.encode_plus(
        human_1,
        padding=True,
        truncation=True,
        return_tensors="pt"
        )
_, logits = sbert_model(**tokens_human)
tokens_machine = tokenizer.encode_plus(
        machine_1,
        padding=True,
        truncation=True,
        return_tensors="pt"
        )
_, logits1 = sbert_model(**tokens_machine)
inputs = sbert_model.tokenize(machine_1)
input_ids = inputs['input_ids'].to(model.device)
input_attention_mask = inputs['attention_mask'].to(model.device)
inputs_final = {'input_ids' : input_ids, 'attention_mask':input_attention_mask}

# model with head
model_with_head = model_with_head.to(model.device)
logits1 = model_with_head(inputs_final)
print(logits)
#loss = criterion(logits, labels)
print(F.sigmoid(logits1).squeeze(1)[0])
print(F.sigmoid(logits1).squeeze(1).sum())
print(F.sigmoid(logits1).squeeze(1).mean())
prob = F.sigmoid(logits).squeeze(1).sum()
prob1 = F.sigmoid(logits1).squeeze(1).sum()

print("H : ", prob, "M : ", prob1)

sentence_embeddings = sbert_model.encode(sentences)
query = "I ate pizza and pasta"
query = human_2
query_vec = sbert_model.encode([query])[0]
for sent in sentences:
    sim = cosine(query_vec, sbert_model.encode([sent])[0])
    print("Sentence = ", sent, "; similarity = ", sim, "Vector Shape = ", query_vec.shape)
"""
#for l, t,c in zip(label, text, wc):
    #print(f"Label : {l}: Text : {t}, Count : {c}")
