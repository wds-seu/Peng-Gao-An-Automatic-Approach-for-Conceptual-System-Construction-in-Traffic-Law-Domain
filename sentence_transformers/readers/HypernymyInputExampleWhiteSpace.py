from typing import Union, List


class HypernymyInputExampleWhiteSpace:
    """
    Structure for one input example with texts, the label and a unique id
    """
    def __init__(self, guid: str = '', texts: List[str] = None, texts_tokenized: List[List[int]] = None, label: Union[int, float] = 0, phrase_list: List[str] = None):
        """
        Creates one InputExample with the given texts, guid and label


        :param guid
            id for the example
        :param texts
            the texts for the example. Note, str.strip() is called on the texts
        :param texts_tokenized
            Optional: Texts that are already tokenized. If texts_tokenized is passed, texts must not be passed.
        :param label
            the label for the example
        :param phrase_list
            the phrase list in the form of [phrase1_str, phrase2_str]
        """
        self.guid = guid
        self.texts = texts
        # this has been changed
        self.texts_tokenized = texts_tokenized
        self.label = label
        if phrase_list is not None:
            self.phrase_list = []

            for phrase in phrase_list:
                self.phrase_list.append(phrase.strip())
        else:
            self.phrase_list = None
        self.phrase_index = None

    def __str__(self):
        return "<InputExample> label: {}, texts: {}".format(str(self.label), "; ".join(self.texts))
