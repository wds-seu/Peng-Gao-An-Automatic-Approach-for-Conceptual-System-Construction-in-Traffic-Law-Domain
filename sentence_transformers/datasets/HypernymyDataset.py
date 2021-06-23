from torch.utils.data import Dataset
from typing import List, Union
import torch
from .. import SentenceTransformer
from ..readers.HypernymyInputExample import HypernymyInputExample
from ..readers.HypernymyInputExampleWhiteSpace import HypernymyInputExampleWhiteSpace


class HypernymyDataset(Dataset):
    """
    Dataset for smart batching, that is each batch is only padded to its longest sequence instead of padding all
    sequences to the max length.
    The SentenceBertEncoder.smart_batching_collate is required for this to work.
    SmartBatchingDataset does *not* work without it.
    """
    def __init__(self,
                 examples: List[Union[HypernymyInputExample, HypernymyInputExampleWhiteSpace]],
                 model: SentenceTransformer
                 ):
        """
        Create a new SentencesDataset with the tokenized texts and the labels as Tensor

        :param examples
            A list of sentence.transformers.readers.InputExample
        :param model:
            SentenceTransformerModel
        """
        self.model = model
        self.examples = examples
        self.label_type = torch.long if isinstance(self.examples[0].label, int) else torch.float

    @classmethod
    def find_phrase_index(cls, tokenized_list: List[int] = None, phrase_list: List[int] = None):
        """
        Find phrase index in the sentence list

        :param tokenized_list: [word1_id, word2_id,...]
        :param phrase_list:  [word1_id, word2_id, ...]
        :return: [[phrase_start_word_id_list], [phrase_end_word_id_list]]
        """
        phrase_index_list = [[], []]
        if len(phrase_list) == 0:
            return None
        indices = [i for i, x in enumerate(tokenized_list) if x == phrase_list[0]]

        if len(indices) == 0:
            return None

        for i in indices:
            start_index = i
            end_index = i

            for id in phrase_list:
                if id != tokenized_list[end_index]:
                    break
                else:
                    end_index += 1
            if end_index - start_index == len(phrase_list):
                phrase_index_list[0].append(start_index)
                phrase_index_list[1].append(end_index)
        if len(phrase_index_list[0]) == 0 or len(phrase_index_list[1]) == 0:
            return None
        else:
            return phrase_index_list

    @classmethod
    def find_xlnet_phrase_index(cls, tokenized_list: List[int] = None, phrase_list: List[int] = None):
        """
        Find phrase index in the sentence list

        :param tokenized_list: [word1_id, word2_id,...]
        :param phrase_list:  [word1_id, word2_id, ...]
        :return: [[phrase_start_word_id_list], [phrase_end_word_id_list]]
        """
        phrase_index_list = [[], []]
        if phrase_list[0] == 19:
            indices = [i for i, x in enumerate(tokenized_list) if x == phrase_list[1]]

            if len(indices) == 0:
                return None

            for i in indices:
                start_index = i
                end_index = i

                for idx in phrase_list[1:]:
                    if idx != tokenized_list[end_index]:
                        break
                    else:
                        end_index += 1
                if end_index - start_index == len(phrase_list)-1:
                    phrase_index_list[0].append(start_index)
                    phrase_index_list[1].append(end_index)
        else:
            indices = [i for i, x in enumerate(tokenized_list) if x == phrase_list[0]]

            if len(indices) == 0:
                return None

            for i in indices:
                start_index = i
                end_index = i

                for idx in phrase_list:
                    if idx != tokenized_list[end_index]:
                        break
                    else:
                        end_index += 1
                if end_index - start_index == len(phrase_list):
                    phrase_index_list[0].append(start_index)
                    phrase_index_list[1].append(end_index)
        if len(phrase_index_list[0]) == 0 or len(phrase_index_list[1]) == 0:
            return None
        else:
            return phrase_index_list

    # examples[item].phrase_index in the form of
    # [phrase1_start_index_list, phrase1_end_index_list, phrase2_start_index_list, phrase2_end_index_list]
    def __getitem__(self, item):
        label = torch.tensor(self.examples[item].label, dtype=self.label_type)
        if self.examples[item].texts_tokenized is None:
            tokenized = []
            phrase_index = []

            # this has been changed
            for i, text in enumerate(self.examples[item].texts):
                phrase_str = self.examples[item].phrase_list[i]
                # print('text:', text)
                # print('phrase:', phrase_str)
                tokenized_list = self.model.tokenize(text.strip())
                phrase_list = self.model.tokenize(phrase_str.strip())
                #
                # print('tokenized list:', tokenized_list)
                # print('phrase list:', phrase_list)
                tokenized.append(tokenized_list)
                phrase_index_list = self.find_phrase_index(tokenized_list, phrase_list)
                # print('phrase index id list:', phrase_index_list)
                if phrase_index_list is None:
                    # raise ValueError('Current sentence do not find phrase index')
                    # print('text:', text)
                    # print('phrase:', phrase_str)
                    # print('tokenized list:', tokenized_list)
                    # print('phrase list:', phrase_list)
                    phrase_index_list = [[], []]
                    phrase_index_list[0].append(0)
                    phrase_index_list[1].append(len(tokenized_list))

                phrase_index.extend(phrase_index_list)
            self.examples[item].texts_tokenized = tokenized
            self.examples[item].phrase_index = phrase_index
        else:
            tokenized = []
            phrase_index = []

            for i, text in enumerate(self.examples[item].texts_tokenized):
                phrase_str = self.examples[item].phrase_list[i]
                phrase_list = self.model.tokenize(phrase_str.strip())
                tokenized.append(self.examples[item].texts_tokenized)
                phrase_index_list = self.find_phrase_index(self.examples[item].texts_tokenized, phrase_list)
                if phrase_index_list is None:
                    phrase_index_list = [[], []]
                    phrase_index_list[0].append(0)
                    phrase_index_list[1].append(len(self.examples[item].texts_tokenized))
                phrase_index.extend(phrase_index_list)
            self.examples[item].phrase_index = phrase_index
            # tokenized = []
            # phrase_index = []
            #
            # for i, text in enumerate(self.examples[item].texts_tokenized):
            #     phrase_str = self.examples[item].phrase_list[i]
            #     print('text:', text)
            #     print('phrase:', phrase_str)
            #     tokenized_list = self.model.tokenize(text.strip())
            #     phrase_list = self.model.tokenize(phrase_str.strip())
            #
            #     print('tokenized list:', tokenized_list)
            #     print('phrase list:', phrase_list)
            #     tokenized.append(tokenized_list)
            #     phrase_index_list = self.find_phrase_index(tokenized_list, phrase_list)
            #     # print('phrase index id list:', phrase_index_list)
            #     if phrase_index_list is None:
            #         raise ValueError('Current sentence do not find phrase index')
            #     phrase_index.extend(phrase_index_list)
            # self.examples[item].texts_tokenized = tokenized
            # self.examples[item].phrase_index = phrase_index

        return self.examples[item].texts_tokenized, label, self.examples[item].phrase_index

    def __len__(self):
        return len(self.examples)
