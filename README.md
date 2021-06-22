# Peng-Gao-Conceptual-System-Construction

# Requirements

conda env: `/penngao_conda_environment.yaml`

pip package: `/penngao_pip_packages.txt`

# Datasets

HITT-h: `/examples/training/hypernymy/datasets/hypernymydetection.tsv.gz`

HITT-a: `/examples/training/attribute/datasets/attributedetection.tsv.gz`

HITT-m: `/examples/training/multirelation/datasets/multi-relation-detection-detection.tsv.gz`

Statistical information of datasets is below:

| Datasets | Train  |  Dev   |  Test  |
| :------: | :----: | :----: | :----: |
|  HITT-h  | 18,847 | 6,302  | 6,292  |
|  HITT-a  | 40,412 | 13,449 | 13,451 |
|  HITT-m  | 19,100 | 6,120  | 6,515  |

# Main File

Hypernymy Detection: `/examples/training/hypernymy/training_hypernymy_benchmark.py`

Concept Attribute Detection: `/examples/training/attribute/training_attribute_benchmark.py`

Multi-relation Detection: `/examples/training/multirelation/training_multi_relation_benchmark.py`

# Run

```
python training_*_benchmark.py
```

# Results

Binary relation detection:

| Model    | Dataset | Accuracy  | Precision | Recall    | F1        |
| -------- | ------- | --------- | --------- | --------- | --------- |
| D-Tensor | HITT-h  | 87.78     | 74.88     | 61.56     | 67.45     |
| D-Tensor | HITT-a  | 83.27     | 70.15     | 60.18     | 65.38     |
| Bran     | HITT-h  | 91.52     | 82.31     | 79.68     | 81.32     |
| Bran     | HITT-a  | 85.34     | 71.25     | 65.48     | 68.56     |
| U_Teal   | HITT-h  | 78.47     | 41.55     | 9.38      | 15.30     |
| U_Teal   | HITT-a  | 77.36     | 40.15     | 10.16     | 16.20     |
| S_Teal   | HITT-h  | 90.85     | 87.03     | 84.31     | 85.56     |
| S_Teal   | HITT-a  | 89.90     | 74.22     | 73.44     | 73.83     |
| AS_Teal  | HITT-h  | **93.36** | **88.67** | 86.22     | 87.89     |
| AS_Teal  | HITT-a  | **92.92** | **80.89** | 79.60     | **79.70** |
| CEE      | HITT-h  | 92.34     | 85.25     | 83.56     | 84.46     |
| CEE      | HITT-a  | 88.56     | 72.17     | 70.86     | 71.56     |
| Ours     | HITT-h  | 93.00     | 87.88     | **89.79** | **88.18** |
| Ours     | HITT-a  | 91.09     | 77.66     | **81.02** | 79.31     |

> D-Tensor: Dual tensor model for detecting asymmetric lexicosemantic relations. EMNLP 2017
>
> Bran: Simultaneously self-attending to all mentions for full-abstract biological relation extraction. NAACL 2018
>
> Teal: Improving hypernymy prediction via taxonomy enhanced adversarial learning. AAAI 2019
>
> CCE: Learning Conceptual-Contextual Embeddings for Medical Text. AAAI 2020

Multi-relation detection:

| Model    | Dataset | Accuracy  | Macro_p   | Macro_R   | Macro_F1  |
| -------- | ------- | --------- | --------- | --------- | --------- |
| D-Tensor | HITT-m  | 75.30     | 76.52     | 73.34     | 74.78     |
| Bran     | HITT-m  | 78.89     | 79.32     | 75.18     | 77.56     |
| CCE      | HITT-m  | 80.12     | 65.89     | 75.30     | 69.53     |
| Ours     | HITT-m  | **81.57** | **82.23** | **81.56** | **81.80** |

# Reference

[SentenceTransformers]([SentenceTransformers Documentation — Sentence-Transformers documentation (sbert.net)](https://www.sbert.net/))

> refer to [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)(EMNLP 2019)
