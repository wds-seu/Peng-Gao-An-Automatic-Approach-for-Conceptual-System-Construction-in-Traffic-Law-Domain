from . import SentenceEvaluator
import logging
import os
import csv
import torch
from torch import nn, Tensor
from sklearn.metrics import average_precision_score, accuracy_score, f1_score, recall_score,precision_score
import numpy as np
from typing import List, Union
from ..readers import HypernymyInputExample, HypernymyInputExampleWhiteSpace


class MultiLabelClassificationEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on its accuracy, precision, recall and f1 on a labeled dataset

    :param sentences1: The first column of sentences
    :param sentences2: The second column of sentences
    :param labels: labels[i] is the label for the pair (sentences1[i], sentences2[i]). Must be 0 or 1
    :param name: Name for the output
    :param phrases1: [phrase_str,...]
    :param phrases2: [phrase_Str,...]
    :param batch_size: Batch size used to compute embeddings
    :param show_progress_bar: If true, prints a progress bar
    """

    def __init__(self,
                 sentences1: List,
                 sentences2: List,
                 labels: List[int],
                 seq_len: int = 128,
                 num_labels: int = 2,
                 name: str = '',
                 sentence_embedding_dimension: int = 768,
                 phrases1: List = None,
                 phrases2: List = None,
                 batch_size: int = 32,
                 show_progress_bar: bool = False):
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.phrases1 = phrases1
        self.phrases2 = phrases2
        self.labels = labels
        self.seq_len = seq_len
        self.num_labels = num_labels
        self.sentence_embedding_dimension = sentence_embedding_dimension

        assert len(self.sentences1) == len(self.sentences2)
        assert len(self.sentences1) == len(self.labels)
        assert len(self.sentences1) == len(self.phrases1)
        assert len(self.sentences1) == len(self.phrases2)

        for label in labels:
            assert (label == 0 or label == 1 or label == 2 or label == 3 or label == 4)

        self.name = name
        self.batch_size = batch_size
        if show_progress_bar is None:
            show_progress_bar = (logging.getLogger().getEffectiveLevel() == logging.INFO or logging.getLogger().getEffectiveLevel() == logging.DEBUG)
        self.show_progress_bar = show_progress_bar

        self.csv_file = "multi_classification_evaluation" + ("_"+name if name else '') + "_measurement.csv"
        self.csv_headers = ["epoch", "steps", "Accuracy",
                            "Macro precision", "Macro recall", "Macro f1-score",
                            "Micro precision", "Micro recall", "Micro f1-score",
                            "Average Precision"]
        self.result_file = "multi_classification_evaluation" + ("_" + name if name else '') + "_result.csv"
        self.result_headers = ["phrase1", "phrase2", "label", "predict", "sentence1", "sentence2"]

    @classmethod
    def from_input_examples(cls,
                            examples: List[Union[HypernymyInputExample, HypernymyInputExampleWhiteSpace]],
                            seq_len: int = 128,
                            num_labels: int = 2,
                            **kwargs):
        sentences1 = []
        sentences2 = []
        phrase1 = []
        phrase2 = []
        scores = []

        for example in examples:
            sentences1.append(example.texts[0])
            sentences2.append(example.texts[1])
            phrase1.append(example.phrase_list[0])
            phrase2.append(example.phrase_list[1])
            scores.append(example.label)
        return cls(sentences1,
                   sentences2,
                   seq_len=seq_len,
                   num_labels=num_labels,
                   phrases1=phrase1,
                   phrases2=phrase2,
                   labels=scores,
                   show_progress_bar=False,
                   **kwargs)

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:

        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}:"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps:"
        else:
            out_txt = ":"

        logging.info("Multi Accuracy Evaluation of the model on " + self.name + " dataset" + out_txt)
        embeddings1 = model.encode(self.sentences1,
                                   batch_size=self.batch_size,
                                   show_progress_bar=self.show_progress_bar,
                                   convert_to_numpy=True,
                                   phrases=self.phrases1,
                                   is_pretokenized=False)
        embeddings2 = model.encode(self.sentences2,
                                   batch_size=self.batch_size,
                                   show_progress_bar=self.show_progress_bar,
                                   convert_to_numpy=True,
                                   phrases=self.phrases2,
                                   is_pretokenized=False)

        # print('embedding1 size:', embeddings1.shape)
        # print('embedding2 size:', embeddings2.shape)

        bilinear_W = model.get_bilinear_W().detach()
        bilinear_W = bilinear_W.cpu().numpy()

        # bilinear product
        new_batch_size = len(embeddings1)
        lin = np.matmul(embeddings1,
                        np.reshape(bilinear_W,
                                   (self.sentence_embedding_dimension, -1)))
        bilin = np.matmul(np.reshape(lin, (new_batch_size,
                                           self.num_labels,
                                           -1)),
                          np.transpose(embeddings2))
        result = self.softmax(bilin, axis=1)
        bilin = np.reshape(result, (new_batch_size, new_batch_size, self.num_labels))
        predictions_list = self.logsumexp(bilin, 1)

        predictions = np.argmax(predictions_list, 1)
        labels = np.asarray(self.labels)

        file_output_data = [epoch, steps]

        # accuracy
        accuracy = accuracy_score(labels, predictions)
        logging.info('Accuracy:\t{:.4f}'.format(accuracy))
        file_output_data.append(accuracy)

        # Macro metric
        Macro_precision = precision_score(labels, predictions, average='macro')
        logging.info('Macro Precision:\t{:.4f}'.format(Macro_precision))
        file_output_data.append(Macro_precision)

        Macro_recall = recall_score(labels, predictions, average='macro')
        logging.info('Macro Recall:\t{:.4f}'.format(Macro_recall))
        file_output_data.append(Macro_recall)

        Macro_f1 = f1_score(labels, predictions, average='macro')
        logging.info('Macro F1:\t{:.4f}'.format(Macro_f1))
        file_output_data.append(Macro_f1)

        # Micro metric
        Micro_precision = precision_score(labels, predictions, average='micro')
        logging.info('Micro Precison:\t{:.4f}'.format(Micro_precision))
        file_output_data.append(Micro_precision)

        Micro_recall = recall_score(labels, predictions, average='micro')
        logging.info('Micro Recall:\t{:.4f}'.format(Micro_recall))
        file_output_data.append(Micro_recall)

        Micro_f1 = f1_score(labels, predictions, average='micro')
        logging.info('Micro F1:\t{:.4f}'.format(Micro_f1))
        file_output_data.append(Micro_f1)

        if output_path is not None:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, mode='w', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow(file_output_data)
            else:
                with open(csv_path, mode='a', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(file_output_data)
            if self.name == 'test':
                result_path = os.path.join(output_path, self.result_file)
                self.write_result_csv(labels=labels,
                                      predictions=predictions,
                                      result_path=result_path)

        return accuracy

    def softmax(self, X, theta=1.0, axis=None):
        """
        Compute the softmax of each element along an axis of X.
        :param X: ND-Array
        :param theta:(optional) float parameter, used as a multiplier prior
            to exponentiation. Default = 1.0
        :param axis: (optional) axis to compute values along. Default is the
            first non-singleton axis.
        :return: an array the same size as X. The result will sum to 1 along
            the specified axis.
        """
        y = np.atleast_2d(X)

        if axis is None:
            axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

        y = y * float(theta)
        y = y - np.expand_dims(np.max(y, axis=axis), axis)
        y = np.exp(y)
        ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)
        p = y / ax_sum
        if len(X.shape) == 1:
            p = p.flatten()

        return p

    def logsumexp(self, x, dim=None, keepdim=False):
        if dim is None:
            x, dim = x.view(-1), 0
        xm = np.max(x, dim, keepdims=True)
        x = np.where(
            (xm == float('inf')) | (xm == float('-inf')),
            xm,
            xm + np.log(np.sum(np.exp(x-xm), dim, keepdims=True))
        )

        return x if keepdim else x.squeeze(dim)

    def write_result_csv(self, labels, predictions, result_path):
        assert len(labels) == len(predictions)
        assert len(self.phrases1) == len(self.phrases2)
        assert len(self.sentences1) == len(self.sentences2)
        assert len(labels) == len(self.phrases1)
        assert len(labels) == len(self.sentences1)

        rows = list(zip(self.phrases1, self.phrases2,
                        labels, predictions,
                        self.sentences1, self.sentences2))

        for i in range(len(rows)-1):
            if not os.path.isfile(result_path):
                with open(result_path, mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.result_headers)
                    writer.writerow(rows[i])
            else:
                with open(result_path, mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(rows[i])