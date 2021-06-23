"""
This example runs a BiLSTM after the word embedding lookup. The output of the BiLSTM is than pooled,
for example with max-pooling (which gives a system like InferSent) or with mean-pooling.

Note, you can also pass BERT embeddings to the BiLSTM.
"""
from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses
from sentence_transformers import HypernymyDataset, LoggingHandler, SentenceTransformer
from sentence_transformers.evaluation import BinaryHypernymyClassificationEvaluator, EmbeddingSimilarityEvaluator
from sentence_transformers.readers import *
import logging
import gzip
import csv
from datetime import datetime

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

# Read the dataset
batch_size = 4
hypernymy_dataset_path = 'datasets/hypernymydetection.tsv.gz'
model_save_path = 'output/training_hypernymy_bilstm-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")



# Map tokens to traditional word embeddings like GloVe
word_embedding_model = models.WordEmbeddings.from_text_file('datasets/chinese.w2c.300d.iter5')

lstm = models.LSTM(word_embedding_dimension=word_embedding_model.get_word_embedding_dimension(), hidden_dim=1024)

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(lstm.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)


model = SentenceTransformer(modules=[word_embedding_model, lstm, pooling_model])


# Convert the dataset to a DataLoader ready for training
logging.info("Read hypernymy train dataset")

train_samples = []
dev_samples = []
test_samples = []
with gzip.open(hypernymy_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)

    for row in reader:
        label = int(row['label'])
        inp_example = HypernymyInputExampleWhiteSpace(texts=[row['sentence1'], row['sentence2']],
                                                      label=label,
                                                      phrase_list=[row['phrase1'], row['phrase2']])
        if row['split'] == 'dev':
            dev_samples.append(inp_example)
        elif row['split'] == 'test':
            test_samples.append(inp_example)
        else:
            train_samples.append(inp_example)

train_data = HypernymyDataset(train_samples, model)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
train_loss = losses.ContrastiveLoss(model=model)

logging.info("Read Hypernymy dev dataset")
evaluator = BinaryHypernymyClassificationEvaluator.from_input_examples(dev_samples, name='hypernymy-dev')

# Configure the training
num_epochs = 10
warmup_steps = math.ceil(len(train_data) * num_epochs / batch_size * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          warmup_steps=warmup_steps,
          evaluation_steps=1000,
          output_path=model_save_path
          )

##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################

model = SentenceTransformer(model_save_path)
evaluator = BinaryHypernymyClassificationEvaluator.from_input_examples(test_samples, name='hypernymy-test')
evaluator(model=model, output_path=model_save_path)
