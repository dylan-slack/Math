### Install

`Python>=3.9`

Requirements
```shell
python = "^3.9"
transformers = "^4.27.3"
datasets = "^2.10.1"
torch = "^2.0.0"
scikit-learn = "^1.2.2"
```

Training Command

```shell
python train.py -m "google/flan-t5-large" --train --overwrite-cache -b 2 --accum 4
```