# Hyperbolic Knowledge Graph Embedding 

This code is the official PyTorch implementation of [Low-Dimensional Hyperbolic Knowledge Graph Embeddings](https://arxiv.org/abs/2005.00545) [6] as well as multiple state-of-the-art KG embedding models which can be trained for the link prediction task. A Tensorflow implementation is also available at: [https://github.com/tensorflow/neural-structured-learning/tree/master/research/kg_hyp_emb](https://github.com/tensorflow/neural-structured-learning/tree/master/research/kg_hyp_emb)

## Library Overview

This implementation includes the following models:

#### Complex embeddings:

*   Complex [1]
*   Complex-N3 [2]
*   RotatE (without self-adversarial sampling) [3]

#### Euclidean embeddings:

*   CTDecomp [2]
*   TransE [4]
*   MurE [5]
*   RotE [6]
*   RefE [6]
*   AttE [6]

#### Hyperbolic embeddings:

*   RotH [6]
*   RefH [6]
*   AttH [6]

## Installation

First, create a python 3.7 environment and install dependencies:

```bash
virtualenv -p python3.7 hyp_kg_env
source hyp_kg_env/bin/activate
pip install -r requirements.txt
```

Then, set environment variables and activate your environment:

```bash
source set_env.sh
```

## Datasets

Download and pre-process the datasets:

```bash
source datasets/download.sh
python datasets/process.py
```

## Usage

To train and evaluate a KG embedding model for the link prediction task, use the run.py script:

```bash
usage: run.py [-h] [--dataset {FB15K,WN,WN18RR,FB237,YAGO3-10}]
              [--model {TransE,CP,MurE,RotE,RefE,AttE,RotH,RefH,AttH,ComplEx,RotatE}]
              [--regularizer {N3,N2}] [--reg REG]
              [--optimizer {Adagrad,Adam,SGD,SparseAdam,RSGD,RAdam}]
              [--max_epochs MAX_EPOCHS] [--patience PATIENCE] [--valid VALID]
              [--rank RANK] [--batch_size BATCH_SIZE]
              [--neg_sample_size NEG_SAMPLE_SIZE] [--dropout DROPOUT]
              [--init_size INIT_SIZE] [--learning_rate LEARNING_RATE]
              [--gamma GAMMA] [--bias {constant,learn,none}]
              [--dtype {single,double}] [--double_neg] [--debug] [--multi_c]

Knowledge Graph Embedding

optional arguments:
  -h, --help            show this help message and exit
  --dataset {FB15K,WN,WN18RR,FB237,YAGO3-10}
                        Knowledge Graph dataset
  --model {TransE,CP,MurE,RotE,RefE,AttE,RotH,RefH,AttH,ComplEx,RotatE}
                        Knowledge Graph embedding model
  --regularizer {N3,N2}
                        Regularizer
  --reg REG             Regularization weight
  --optimizer {Adagrad,Adam,SparseAdam}
                        Optimizer
  --max_epochs MAX_EPOCHS
                        Maximum number of epochs to train for
  --patience PATIENCE   Number of epochs before early stopping
  --valid VALID         Number of epochs before validation
  --rank RANK           Embedding dimension
  --batch_size BATCH_SIZE
                        Batch size
  --neg_sample_size NEG_SAMPLE_SIZE
                        Negative sample size, -1 to not use negative sampling
  --dropout DROPOUT     Dropout rate
  --init_size INIT_SIZE
                        Initial embeddings' scale
  --learning_rate LEARNING_RATE
                        Learning rate
  --gamma GAMMA         Margin for distance-based losses
  --bias {constant,learn,none}
                        Bias type (none for no bias)
  --dtype {single,double}
                        Machine precision
  --double_neg          Whether to negative sample both head and tail entities
  --debug               Only use 1000 examples for debugging
  --multi_c             Multiple curvatures per relation
```

## Examples 

We provide example scripts with hyper-parameters for WN18RR in the examples/ folder. For dimensions 32 and 500, these models should achieve the following test MRRs:

|   model    | rank |  MRR  | H@10 |
|------------|------|-------|------|
| ComplEx-N3 |  32  | .407  | .449 |
| ComplEx-N3 | 500  | .477  | .572 |
|    RotE    |  32  | .455  | .527 |
|    RotE    | 500  | .494  | .587 |
|    RotH    |  32  | .473  | .551 |
|    RotH    | 500  | .489  | .581 |

## New models

To add a new (complex/hyperbolic/Euclidean) Knowledge Graph embedding model, implement the corresponding query embedding under models/, e.g.:

```
def get_queries(self, queries):
    head_e = self.entity(queries[:, 0])
    rel_e = self.rel(queries[:, 1])
    lhs_e = ### Do something here ###
    lhs_biases = self.bh(queries[:, 0])
    return lhs_e, lhs_biases
```

## Citation

If you use the codes, please cite the following paper [6]:

```
@inproceedings{chami2020low,
  title={Low-Dimensional Hyperbolic Knowledge Graph Embeddings},
  author={Chami, Ines and Wolf, Adva and Juan, Da-Cheng and Sala, Frederic and Ravi, Sujith and R{\'e}, Christopher},
  booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  pages={6901--6914},
  year={2020}
}
```

## References

[1] Trouillon, Théo, et al. "Complex embeddings for simple link prediction."
International Conference on Machine Learning. 2016.

[2] Lacroix, Timothee, et al. "Canonical Tensor Decomposition for Knowledge Base
Completion." International Conference on Machine Learning. 2018.

[3] Sun, Zhiqing, et al. "Rotate: Knowledge graph embedding by relational
rotation in complex space." International Conference on Learning
Representations. 2019.

[4] Bordes, Antoine, et al. "Translating embeddings for modeling
multi-relational data." Advances in neural information processing systems. 2013.

[5] Balažević, Ivana, et al. "Multi-relational Poincaré Graph Embeddings."
Advances in neural information processing systems. 2019.

[6] Chami, Ines, et al. "Low-Dimensional Hyperbolic Knowledge Graph Embeddings."
Annual Meeting of the Association for Computational Linguistics. 2020.

Some of the code was forked from the original ComplEx-N3 implementation which can be found at: [https://github.com/facebookresearch/kbc](https://github.com/facebookresearch/kbc)

