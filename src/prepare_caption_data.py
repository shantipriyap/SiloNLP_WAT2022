import os

import pandas as pd


def prepare_caption_data(fpath: str, save_path: str, trglang: str):
    """Prepare dataframe and save to file"""
    src_col_name = 5
    trg_col_name = 6
    df = pd.read_csv(fpath, encoding='utf-8', sep='\t', header=None)
    df = df[[src_col_name, trg_col_name]]
    df.rename(columns={src_col_name: 'en', trg_col_name: trglang}, inplace=True)
    df.to_csv(save_path, sep='\t', encoding='utf-8', index=False)


if __name__ == "__main__":

    for target_lang, full_lang, folder_name, challenge_prefix in zip(
            ['hi', 'bn', 'ml'],
            ['hindi', 'bengali', 'malayalam'],
            ['hvg', 'bvg', 'mvg'],
            ['challenge-test-set', 'challenge-test-set', 'chtest']
    ):
        inp_data_dir = os.path.join(f'../data/raw/{folder_name}')
        out_data_dir = os.path.join(f'../data/prepared/{folder_name}')

        visual_genome_data_paths = {
            'train': os.path.join(inp_data_dir, f'{full_lang}-visual-genome-train.txt'),
            'dev': os.path.join(inp_data_dir, f'{full_lang}-visual-genome-dev.txt'),
            'test': os.path.join(inp_data_dir, f'{full_lang}-visual-genome-test.txt'),
            'challenge': os.path.join(inp_data_dir, f'{full_lang}-visual-genome-{challenge_prefix}.txt'),
        }

        # prepare caption data
        os.makedirs(out_data_dir, exist_ok=True)
        for split in ['train', 'dev', 'test', 'challenge']:
            prepare_caption_data(
                fpath=visual_genome_data_paths[split],
                save_path=os.path.join(out_data_dir, f'{split}.en.{target_lang}.tsv'),
                trglang=target_lang,
            )
