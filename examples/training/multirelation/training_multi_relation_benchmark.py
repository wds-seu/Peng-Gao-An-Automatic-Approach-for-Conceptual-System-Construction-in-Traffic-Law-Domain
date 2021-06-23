from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer, HypernymyDataset, LoggingHandler, losses, models
from sentence_transformers.evaluation import MultiClassificationEvaluator
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
multi_relation_dataset_path = 'datasets/multi-relation-detection-split.tsv.gz'
# multi_relation_dataset_path = 'datasets/multi-relation-detection-same-train-dev.tsv.gz'
# multi_relation_dataset_path = 'datasets/multi-relation-detection-without-0-label.tsv.gz'

# You can specify any huggingface/transformers pre-trained model here
model_name = sys.argv[1] if len(sys.argv) > 1 else 'bert-base-chinese'

# Read the dataset
train_batch_size = 4
num_epochs = 3
num_labels = 5

# model parameter name
pooling_name = 'mean_pooling_'

# phrase_attention_name = ''
phrase_attention_name = 'phrase_attention_'

# tensor_name = ''
tensor_name = 'tensor_'

model_save_path = 'output/training_multi_relation_benchmark_' + \
                  model_name.replace("/", "_") + '_' + \
                  pooling_name + \
                  phrase_attention_name + \
                  tensor_name + \
                  datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

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
logging.info("Read multi relation train dataset")
train_samples = []
dev_samples = []
test_samples = []
train_label_0_total_count = 3820
rate = 0.8
train_label_0_count, train_label_1_count, train_label_2_count, \
    train_label_3_count, train_label_4_count = 0, 0, 0, 0, 0
dev_label_0_count, dev_label_1_count, dev_label_2_count, \
    dev_label_3_count, dev_label_4_count = 0, 0, 0, 0, 0
test_label_0_count, test_label_1_count, test_label_2_count, \
    test_label_3_count, test_label_4_count = 0, 0, 0, 0, 0

with gzip.open(multi_relation_dataset_path, 'rt', encoding='utf8') as fIn:
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
                dev_label_0_count += 1
            elif label == 1:
                dev_label_1_count += 1
            elif label == 2:
                dev_label_2_count += 1
            elif label == 3:
                dev_label_3_count += 1
            elif label == 4:
                dev_label_4_count += 1
        elif row['split'] == 'test':
            test_samples.append(inp_example)
            if label == 0:
                test_label_0_count += 1
            elif label == 1:
                test_label_1_count += 1
            elif label == 2:
                test_label_2_count += 1
            elif label == 3:
                test_label_3_count += 1
            elif label == 4:
                test_label_4_count += 1
        else:
            if label == 0 and train_label_0_count < int(rate*train_label_0_total_count):
                train_label_0_count += 1
                train_samples.append(inp_example)
            elif label == 1 and train_label_1_count < int(rate*train_label_0_total_count):
                train_label_1_count += 1
                train_samples.append(inp_example)
            elif label == 2 and train_label_2_count < int(rate*train_label_0_total_count):
                train_label_2_count += 1
                train_samples.append(inp_example)
            elif label == 3 and train_label_3_count < int(rate*train_label_0_total_count):
                train_label_3_count += 1
                train_samples.append(inp_example)
            elif label == 4 and train_label_4_count < int(rate*train_label_0_total_count):
                train_label_4_count += 1
                train_samples.append(inp_example)
print('train label 0 count:', train_label_0_count)
print('train label 1 count:', train_label_1_count)
print('train label 2 count:', train_label_2_count)
print('train label 3 count:', train_label_3_count)
print('train label 4 count:', train_label_4_count)
print('dev label 0 count:', dev_label_0_count)
print('dev label 1 count:', dev_label_1_count)
print('dev label 2 count:', dev_label_2_count)
print('dev label 3 count:', dev_label_3_count)
print('dev label 4 count:', dev_label_4_count)
print('test label 0 count:', test_label_0_count)
print('test label 1 count:', test_label_1_count)
print('test label 2 count:', test_label_2_count)
print('test label 3 count:', test_label_3_count)
print('test label 4 count:', test_label_4_count)

train_dataset = HypernymyDataset(train_samples,
                                 model)
train_dataloader = DataLoader(train_dataset,
                              shuffle=True,
                              batch_size=train_batch_size)
train_loss = losses.MultiSoftmaxLoss(model=model,
                                     sentence_embedding_dimension=sentence_embedding_dimension,
                                     num_labels=num_labels,
                                     seq_len=max_seq_len)
# train_loss = losses.ContrastiveLoss(model=model)
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

logging.info("Read multi relation dev dataset")
evaluator = MultiClassificationEvaluator.from_input_examples(dev_samples,
                                                             num_labels=num_labels,
                                                             seq_len=max_seq_len,
                                                             name='dev')

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

# Load the stored model and evaluate its performance on multi relation benchmark dataset
bilinear_w_pkl = os.path.join(model_save_path, 'bilinear_w.pkl')
model = SentenceTransformer(model_save_path, bilinear_w_pkl=bilinear_w_pkl)
test_evaluator = MultiClassificationEvaluator.from_input_examples(test_samples,
                                                                  num_labels=num_labels,
                                                                  seq_len=max_seq_len,
                                                                  name='test')
test_evaluator(model=model, output_path=model_save_path)