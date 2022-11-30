# SiloNLP_WAT2022

The repository contains the code/data of team "Silo NLP's" participation at the Workshop on Asian Translation (WAT2022) for the below tasks.
* English->Hindi Multimodal Translation 
* English->Malayalam Multimodal Translation
* English->Bengali Multimodal Translation

## Assumption
The object tags were already extracted using Faster-RCNN and added to the repository for multimodal machine translation (MMT).

## Abbreviations Used 
* Hindi Visual Genome -- HVG
* Malayalam Visual Genome -- MVG
* Bengali Visual Genome -- BVG
* English -- EN
* Hindi -- HI
* Malayalam -- ML
* Bengali -- BN


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
$ python finetune_mbart.py hi multimodal ../data/prepared_object_tags_concat_hvg_flickr/hvg
```

If you are using the code, please cite the associated paper:

@inproceedings{parida-etal-2022-silo,
    title = "Silo {NLP}{'}s Participation at {WAT}2022",
    author = {Parida, Shantipriya  and
      Panda, Subhadarshi  and
      Gr{\"o}nroos, Stig-Arne  and
      Granroth-Wilding, Mark  and
      Koistinen, Mika},
    booktitle = "Proceedings of the 9th Workshop on Asian Translation",
    month = oct,
    year = "2022",
    address = "Gyeongju, Republic of Korea",
    publisher = "International Conference on Computational Linguistics",
    url = "https://aclanthology.org/2022.wat-1.12",
    pages = "99--105",
  }
