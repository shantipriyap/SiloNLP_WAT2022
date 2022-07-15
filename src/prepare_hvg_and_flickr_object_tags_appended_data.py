import os

import pandas as pd

if __name__ == "__main__":
    FLICKR_TRAIN_PATH = os.path.join('../data/flickr8k/flickr8k.synthetic_features.tsv')
    INP_DIR = os.path.join('../data/prepared_object_tags/hvg')
    OUT_DIR = os.path.join('../data/prepared_object_tags_concat_hvg_flickr/hvg')

    os.makedirs(OUT_DIR, exist_ok=True)

    filenames = os.listdir(INP_DIR)
    for filename in filenames:
        df = pd.read_csv(os.path.join(INP_DIR, filename), sep='\t', encoding='utf-8')
        if filename.split('.')[0] == 'train':  # train data
            flickr_df = pd.read_csv(FLICKR_TRAIN_PATH, sep='\t', encoding='utf-8')
            concat_df = pd.concat([df, flickr_df], axis=0)
            concat_df.to_csv(os.path.join(OUT_DIR, filename), sep='\t', encoding='utf-8', index=False)
        else:  # dev, test, challenge data
            df.to_csv(os.path.join(OUT_DIR, filename), sep='\t', encoding='utf-8', index=False)
