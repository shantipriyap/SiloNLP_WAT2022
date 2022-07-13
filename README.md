# WAT2022

The repository contains the code/data for Workshop on Asian Translation (WAT2022).


## Multimodal Tasks 

Here is how to run:

* Put raw HVG data in data/raw/hvg
* Put raw BVG data in data/raw/bvg
* Put raw MVG data in data/raw/mvg

```
$ cd src
```

Prepare caption data for hi, bn, ml

```
$ python prepare_caption_data.py
```

Append object tags for hi, bn, ml

```
$ python append_object_tags.py
```

Finetune mBART for text only translation

```
$ python finetune_mbart.py hi text ../data/prepared/hvg
$ python finetune_mbart.py bn text ../data/prepared/bvg
$ python finetune_mbart.py ml text ../data/prepared/mvg
```

Finetune mBART for multimodal translation

```
$ python finetune_mbart.py hi multimodal ../data/prepared_object_tags/hvg
$ python finetune_mbart.py bn multimodal ../data/prepared_object_tags/bvg
$ python finetune_mbart.py ml multimodal ../data/prepared_object_tags/mvg
```
