import os
from typing import List

import pandas as pd
from tqdm import tqdm


def load_obj_tags(fpath: str) -> pd.DataFrame:
    """Load object tags from file"""
    df = pd.read_csv(fpath, encoding='utf-8', sep='\t')
    return df


def append_object_tags(data_df: pd.DataFrame, image_ids: List[int], object_tags_df: pd.DataFrame, srclang: str,
                       topn: int,
                       save_path: str) -> None:
    """Append object tags to source text"""
    # sanity check
    assert len(data_df) == len(object_tags_df) == len(image_ids)

    src_sents = data_df[srclang].astype(str).to_list()
    obj_tags_dict = {}
    object_tags_df['obj_tags'] = object_tags_df['obj_tags'].fillna('')
    for img_idx, obj_tags in zip(object_tags_df['image_idx'].to_list(),
                                 object_tags_df['obj_tags'].astype(str).to_list()):
        obj_tags_dict[img_idx] = obj_tags
    ordered_obj_tags = [obj_tags_dict[image_idx] for image_idx in image_ids]
    src_sents_with_obj_tags = []
    for src_sent, obj_tags in tqdm(
            zip(src_sents, ordered_obj_tags),
            total=len(data_df)):
        src_sents_with_obj_tags.append(f'{src_sent}##{",".join(obj_tags.split(",")[:topn])}')

    data_df[srclang] = src_sents_with_obj_tags
    data_df.to_csv(save_path, sep='\t', encoding='utf-8', index=False)


if __name__ == "__main__":
    # NOTE: make sure that the object tags are inferred from the image ids and not from the order of the sentences.
    visual_genome_image_order_paths = {
        'train': os.path.join('../data/raw/hvg/hindi-visual-genome-train.txt'),
        'dev': os.path.join('../data/raw/hvg/hindi-visual-genome-dev.txt'),
        'test': os.path.join('../data/raw/hvg/hindi-visual-genome-test.txt'),
        'challenge': os.path.join('../data/raw/hvg/hindi-visual-genome-challenge-test-set.txt'),
    }
    obj_tags_dir = os.path.join('../data/object_tags')
    for target_lang, folder_name in zip(['hi', 'bn', 'ml'], ['hvg', 'bvg', 'mvg']):
        prepared_data_dir = os.path.join(f'../data/prepared/{folder_name}')
        save_dir = os.path.join(f'../data/prepared_object_tags/{folder_name}')
        os.makedirs(save_dir, exist_ok=True)
        visual_genome_data_paths = {split: os.path.join(prepared_data_dir, f'{split}.en.{target_lang}.tsv') for split in
                                    ['train', 'dev', 'test', 'challenge']}
        for split, path in visual_genome_data_paths.items():
            append_object_tags(
                data_df=pd.read_csv(path, sep='\t', encoding='utf-8', na_filter=False),
                image_ids=pd.read_csv(visual_genome_image_order_paths[split], encoding='utf-8', sep='\t', header=None)[
                    0].to_list(),
                object_tags_df=load_obj_tags(os.path.join(obj_tags_dir, f'{split}_obj_tags.tsv')),
                srclang='en',
                topn=10,
                save_path=os.path.join(save_dir, f'{split}.en.{target_lang}.tsv'),
            )
