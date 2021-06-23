from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer, HypernymyDataset, LoggingHandler, losses, models
from sentence_transformers.evaluation import BinaryHypernymyClassificationEvaluator, \
    MultiLabelClassificationEvaluator, MultiClassificationEvaluator
from sentence_transformers.readers import HypernymyInputExample
from datetime import datetime
import logging
import sys
import gzip
import csv
import os

# Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
# print debug information to stdout

# dataset file
# hypernymy_dataset_path = 'datasets/hypernymydetection_lexical.tsv.gz'
hypernymy_dataset_path = 'datasets/hypernymydetection.tsv.gz'
# hypernymy_dataset_path = 'datasets/hypernymydetection_new.tsv.gz'
# hypernymy_dataset_path = '../multirelation/datasets/multi-relation-detection.tsv.gz'

# You can specify any huggingface/transformers pre-trained model here
model_name = sys.argv[1] if len(sys.argv) > 1 else 'bert-base-chinese'

# Read the dataset
train_batch_size = 8
num_epochs = 5
num_labels = 2

# model parameter name
pooling_name = 'mean_pooling_'

# phrase_attention_name = ''
phrase_attention_name = 'phrase_attention_'

tensor_name = ''
# tensor_name = 'tensor_'

# model_save_path = 'output/training_hypernymy_lexical_' + \
#                   model_name.replace("/", "_") + '_' + \
#                   pooling_name + \
#                   phrase_attention_name + \
#                   tensor_name + \
#                   datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

model_save_path = 'output/training_hypernymy_benchmark_' + \
                  model_name.replace("/", "_") + '_' + \
                  pooling_name + \
                  phrase_attention_name + \
                  tensor_name + \
                  datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# model_save_path = 'output/training_hypernymy_benchmark_new_' + \
#                   model_name.replace("/", "_") + '_' + \
#                   pooling_name + \
#                   phrase_attention_name + \
#                   tensor_name + \
#                   datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Record start time
start_time = datetime.now()

# Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
word_embedding_model = models.Transformer(model_name)
max_seq_len = word_embedding_model.get_max_seq_length()

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False,
                               pooling_mode_mean_sqrt_len_tokens=False)
token_embeddings_dimension = word_embedding_model.get_word_embedding_dimension()

phrase_attention = models.MultiHeadAttention(token_embeddings_dimension,
                                             token_embeddings_dimension,
                                             num_units=token_embeddings_dimension)
sentence_embedding_dimension = phrase_attention.get_sentence_embedding_dimension()

model = SentenceTransformer(modules=[word_embedding_model,
                                     phrase_attention,
                                     pooling_model])

# Convert the dataset to a DataLoader ready for training
logging.info("Read Hypernymy train dataset")

train_samples = []
dev_samples = []
test_samples = []
train_positive_total_count = 3820
train_negative_total_count = 15027
rate = 0.8
train_positive_count, train_negative_count = 0, 0
dev_positive_count, dev_negative_count = 0, 0
test_positive_count, test_negative_count = 0, 0

with gzip.open(hypernymy_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)

    for row in reader:
        label = int(row['label'])
        inp_example = HypernymyInputExample(texts=[row['sentence1'], row['sentence2']],
                                            label=label,
                                            phrase_list=[row['phrase1'],
                                                         row['phrase2']])
        if row['split'] == 'dev':
            dev_samples.append(inp_example)
            if label == 0:
                dev_negative_count += 1
            else:
                dev_positive_count += 1
        elif row['split'] == 'test':
            test_samples.append(inp_example)
            if label == 0:
                test_negative_count += 1
            else:
                test_positive_count += 1
        else:
            if label == 0 and train_negative_count < int(train_negative_total_count * rate):
                train_negative_count += 1
                train_samples.append(inp_example)
            elif label == 1 and train_positive_count < int(train_positive_total_count * rate):
                train_positive_count += 1
                train_samples.append(inp_example)
print('train positive count:', train_positive_count)
print('train negative count:', train_negative_count)
print('dev positive count:', dev_positive_count)
print('dev negative count:', dev_negative_count)
print('test positive count:', test_positive_count)
print('test negative count:', test_negative_count)

train_dataset = HypernymyDataset(train_samples,
                                 model)
train_dataloader = DataLoader(train_dataset,
                              shuffle=True,
                              batch_size=train_batch_size)
train_loss = losses.ContrastiveLoss(model=model)
# train_loss = losses.CosineSimilarityLoss(model=model)
# train_loss = losses.ContrastiveSoftmaxLoss(model=model,
#                                            sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
#                                            num_labels=2)
# train_loss = losses.SoftmaxLoss(model=model,
#                                 sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
#                                 num_labels=2,
#                                 concatenation_sent_rep=True,
#                                 concatenation_sent_difference=True,
#                                 concatenation_sent_multiplication=False)
# train_loss = losses.MultiLabelSoftmaxLoss(model=model,
#                                           sentence_embedding_dimension=sentence_embedding_dimension,
#                                           num_labels=num_labels,
#                                           seq_len=max_seq_len)
# train_loss = losses.MultiSoftmaxLoss(model=model,
#                                      sentence_embedding_dimension=sentence_embedding_dimension,
#                                      num_labels=num_labels,
#                                      seq_len=max_seq_len)

logging.info("Read Hypernymy dev dataset")
evaluator = BinaryHypernymyClassificationEvaluator.from_input_examples(dev_samples, name='dev')
# evaluator = MultiLabelClassificationEvaluator.from_input_examples(dev_samples,
#                                                                   num_labels=num_labels,
#                                                                   seq_len=max_seq_len,
#                                                                   name='dev')
# evaluator = MultiClassificationEvaluator.from_input_examples(dev_samples,
#                                                              num_labels=num_labels,
#                                                              seq_len=max_seq_len,
#                                                              name='dev')

# Configure the training. We skip evaluation in this example
# 10% of train data for warm-up
warmup_steps = math.ceil(len(train_dataset) * num_epochs / train_batch_size * 0.1)
logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          warmup_steps=warmup_steps,
          evaluation_steps=1000,
          output_path=model_save_path)

# Record train end time
train_end_time = datetime.now()
print('Training time is:{} s'.format((train_end_time-start_time).seconds))

# Load the stored model and evaluate its performance on hypernymy benchmark dataset
bilinear_w_pkl = os.path.join(model_save_path, 'bilinear_w.pkl')
model = SentenceTransformer(model_save_path)
test_evaluator = BinaryHypernymyClassificationEvaluator.from_input_examples(test_samples, name='test')
# test_evaluator = MultiLabelClassificationEvaluator.from_input_examples(test_samples,
#                                                                        num_labels=num_labels,
#                                                                        seq_len=max_seq_len,
#                                                                        name='test')
# test_evaluator = MultiClassificationEvaluator.from_input_examples(test_samples,
#                                                                   num_labels=num_labels,
#                                                                   seq_len=max_seq_len,
#                                                                   name='test')
test_evaluator(model=model, output_path=model_save_path)