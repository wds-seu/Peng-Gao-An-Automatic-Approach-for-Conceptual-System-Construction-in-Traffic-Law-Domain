from torch.utils.data import Dataset
from typing import List, Union
from .. import SentenceTransformer


class HypernymyEncodeDataset(Dataset):
    def __init__(self,
                 sentences: Union[List[str], List[int]],
                 model: SentenceTransformer,
                 phrases: List = None,
                 is_tokenized: bool = True):
        """
        EncodeDataset is used by SentenceTransformer.encode method. It just stores
        the input texts and returns a tokenized version of it.
        """
        self.model = model
        self.sentences = sentences
        self.is_tokenized = is_tokenized
        self.phrases = phrases


    def __getitem__(self, item):
        if self.is_tokenized:
            return self.sentences[item], self.phrases[item]
        else:
            return self.model.tokenize(self.sentences[item]), self.model.tokenize(self.phrases[item])


    def __len__(self):
        return len(self.sentences)
