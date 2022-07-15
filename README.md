# WAT2022

The repository contains the code/data for Workshop on Asian Translation (WAT2022).

## Environment Details

* Pytorch
* sentencepiece
* sacrebleu
* transformers

Note: Based on you environment (e.g. CUDA) select the Pytorch version

Example: 
* Name: torch
* Version: 1.7.1+cu110
* Name: transformers                          
* Version: 4.6.0
* Name: sentencepiece                         
* Version: 0.1.96
* Name: sacrebleu                             
* Version: 1.5.1


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

Finetune mBART for multimodal translation with concatenated HVG and Flickr data

```
$ python prepare_hvg_and_flickr_object_tags_appended_data.py
$ python finetune_mbart.py hi multimodal ../data/$ python finetune_mbart.py hi multimodal ../data/prepared_object_tags_concat_hvg_flickr/hvg
```
