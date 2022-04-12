# CLMLF

This is the repository of *CLMLF:A Contrastive Learning and Multi-Layer Fusion Method for
Multimodal Sentiment Detection* [<a href='http' >pdf</a>] (NAACL 2022 Finding)

<div align=center>
<img src='util\img\CLMLF.svg' width=90% alt=''>
</div>

## Requirements

We give the version of the python package we used, please refer to `requirements.txt`

```
python 3.6
pytorch 1.7.1
transformer 4.10.0
```

## Data

We provide text data in `dataset\data\`. As for images, please download from the link below:

MVSA-*

<a href='http://mcrlab.net/research/mvsa-sentiment-analysis-on-multi-view-social-data/'>http://mcrlab.net/research/mvsa-sentiment-analysis-on-multi-view-social-data/</a>


HFM
<a href='https://github.com/headacheboy/data-of-multimodal-sarcasm-detection'>https://github.com/headacheboy/data-of-multimodal-sarcasm-detection</a>

For example, MVSA-Single's images should be put in `dataset\data\MVSA-single\dataset_image`

## Run code

We provide running scripts as follow:

As for MVSA-Single:
```
sh train-single.sh 0
```

As for MVSA-Mulitple:
```
sh train-mul.sh 0
```

As for HFM:
```
sh train-hfm.sh 0
```

## Citation 

If you find this code useful for your research, please consider citing:

```
@inproceedings{li2022clmlf,
  title={CLMLF:A Contrastive Learning and Multi-Layer Fusion Method for Multimodal Sentiment Detection},
  author={Zhen Li, Bing Xu, Conghui Zhu, Tiejun Zhao},
  booktitle={NAACL2022 Finding},
  year={2022}
}
```