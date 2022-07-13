# -*- coding: utf-8 -*-

"""
Installation:
pip install sentencepiece
pip install sacrebleu
pip install transformers==4.6.0
"""

import math
import os
import random
import sys
import time
from typing import List, Tuple, Dict, Any, Union

import numpy as np
import pandas as pd
import sacrebleu
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm as tqdm
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

assert len(sys.argv) == 4, 'Usage: python finetune_mbart.py LANG MODE DATA_DIR'  # sanity check

SEED = 123
TRGLANG = sys.argv[1]  # choose between hi, bn, ml
BATCH_SIZE = 1
LEARNING_RATE = 0.0001
BEAM_SIZE = 5
EARLY_STOPPING_PATIENCE = 5
#MAX_EPOCHS = 1
MAX_EPOCHS = 30
GENERATE_ONLY = False  # if generate only is set to True, make sure that a trained checkpoint exists
#GENERATE_ONLY = True  # if generate only is set to True, make sure that a trained checkpoint exists
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODE = sys.argv[2]  # choose between "multimodal" and "text". accordingly set the DATA_DIR.
DATA_DIR = os.path.join(sys.argv[3])
#TRAIN_PATH = os.path.join(DATA_DIR, f'train.en.{TRGLANG}.txt')
TRAIN_PATH = os.path.join(DATA_DIR, f'train.en.{TRGLANG}.tsv')
print("TRAIN_PATH:",TRAIN_PATH)
#DEV_PATH = os.path.join(DATA_DIR, f'dev.en.{TRGLANG}.txt')
DEV_PATH = os.path.join(DATA_DIR, f'dev.en.{TRGLANG}.tsv')
#TEST_PATH = os.path.join(DATA_DIR, f'test.en.{TRGLANG}.txt')
TEST_PATH = os.path.join(DATA_DIR, f'test.en.{TRGLANG}.tsv')
#CHALLENGE_PATH = os.path.join(DATA_DIR, f'challenge.en.{TRGLANG}.txt')
CHALLENGE_PATH = os.path.join(DATA_DIR, f'challenge.en.{TRGLANG}.tsv')
MODEL_DIR = os.path.join(f'{TRGLANG}.{MODE}.model')
TRANSLATIONS_DIR = os.path.join(f'{TRGLANG}.{MODE}.translations')
HISTORY_FILE_PATH = os.path.join(f'{TRGLANG}.{MODE}.history.tsv')

# sanity checks
#assert torch.__version__.startswith("1.8")
assert os.path.isfile(TRAIN_PATH)
assert os.path.isfile(DEV_PATH)
assert os.path.isfile(TEST_PATH)
assert os.path.isfile(CHALLENGE_PATH)

"""Load pretrained tokenizer and pretrained model"""
TOKENIZER = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-one-to-many-mmt", src_lang="en_XX",
                                                 tgt_lang=f"{TRGLANG}_IN")

MODEL = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-one-to-many-mmt")


class TranslationDataset(Dataset):
    """Translation dataset class"""

    def __init__(self, data_df: pd.DataFrame, src_lang: str, trg_lang: str, tokenizer):
        self.data_df = data_df
        self.src_lang = src_lang
        self.trg_lang = trg_lang
        self.tokenizer = tokenizer
        self.src_text = data_df[src_lang].astype(str).to_list()
        self.trg_text = data_df[trg_lang].astype(str).to_list()

    def __getitem__(self, idx):
        return self.src_text[idx], self.trg_text[idx]

    def __len__(self):
        return len(self.data_df)

    def collate_fn(self, batch: List[Tuple[str, str]]):
        inputs = self.tokenizer([tup[0] for tup in batch], return_tensors="pt", padding=True)
        with self.tokenizer.as_target_tokenizer():
            targets = self.tokenizer([tup[1] for tup in batch], return_tensors="pt", padding=True).input_ids
        return inputs, targets


def transfer_inputs_to_device(inputs):
    return {k: v.to(DEVICE) for k, v in inputs.items()}


def transfer_targets_to_device(targets):
    return targets.to(DEVICE)


def train(model, iterator, optimizer, curr_epoch: int, max_epochs: int) -> float:
    """Train method"""
    # set model to train mode
    model.train()

    epoch_loss = 0.0

    tqdm_meter = tqdm(
        iterator,
        unit=' batches',
        desc=f'[EPOCH {curr_epoch}/{max_epochs}]',
        leave=False,
    )

    for batch_idx, batch in enumerate(tqdm_meter):
        inputs, targets = batch[0], batch[1]

        # transfer to device
        inputs = transfer_inputs_to_device(inputs)
        targets = transfer_targets_to_device(targets)

        # zero out optimizer
        optimizer.zero_grad()

        # forward pass and compute loss
        loss = model(**inputs, labels=targets)['loss']

        # backward
        loss.backward()

        # # clip grad norm
        # torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        # optimizer step
        optimizer.step()

        epoch_loss += loss.item()

        # update tqdm meter
        tqdm_meter.set_postfix(
            ordered_dict={
                'loss': f'{loss.item():0.4f}',
                'ppl': f'{math.exp(loss.item()):5.4f}',
            }
        )
        tqdm_meter.update()

    return epoch_loss / (batch_idx + 1)


def evaluate(model, iterator, beam_size: int, references: List[str]) -> Dict[str, Union[float, List[str]]]:
    """Generate and evaluate method"""

    # set model to evaluation mode
    model.eval()

    epoch_loss = 0.0

    tqdm_meter = tqdm(
        iterator,
        unit=' batches',
        desc=f'compute eval loss',
        leave=False,
    )

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm_meter):
            inputs, targets = batch[0], batch[1]

            # transfer to device
            inputs = transfer_inputs_to_device(inputs)
            targets = transfer_targets_to_device(targets)

            # forward pass and compute loss
            loss = model(**inputs, labels=targets)['loss']

            epoch_loss += loss.item()

            # update tqdm meter
            tqdm_meter.set_postfix(
                ordered_dict={
                    'loss': f'{loss.item():0.4f}',
                    'ppl': f'{math.exp(loss.item()):5.4f}',
                }
            )
            tqdm_meter.update()

    # generate translations
    translations = generate(model, iterator, beam_size=beam_size)

    # compute bleu score
    bleu_score = sacrebleu.corpus_bleu(sys_stream=translations, ref_streams=[references], force=False).score
    #bleu_score = sacrebleu.corpus_bleu(translations, references, force=False).score

    return {
        'loss': epoch_loss / (batch_idx + 1),
        'translations': translations,
        'bleu_score': bleu_score,
    }


def generate(model, iterator, beam_size: int) -> List[str]:
    """Generate method"""

    # set model to evaluation mode
    model.eval()

    translations = []

    tqdm_meter = tqdm(
        iterator,
        unit=' batches',
        desc=f'translate',
        leave=False,
    )

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm_meter):
            inputs = batch[0]

            # transfer to device
            inputs = transfer_inputs_to_device(inputs)

            # generate
            generated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=TOKENIZER.lang_code_to_id[f"{TRGLANG}_IN"],
                num_beams=beam_size,
            )
            batch_translations = TOKENIZER.batch_decode(generated_tokens, skip_special_tokens=True)
            translations.extend(batch_translations)

    assert len(translations) == batch_idx + 1
    return translations


def seed_everything(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(fpath: str) -> pd.DataFrame:
    """Load data from file to a dataframe"""
    df = pd.read_csv(fpath, encoding='utf-8', sep='\t')
    return df


def epoch_time(start_time, end_time):
    """Compute epoch time"""
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train_till_convergence(
        model,
        train_iter, dev_iter, test_iter, challenge_iter,
        dev_references: List[str], test_references: List[str], challenge_references: List[str],
        lr: float, max_epochs: int,
        model_dir: str, translations_dir: str, early_stopping_patience: int, beam_size: int
) -> Dict[str, Any]:
    # transfer model to device
    model = model.to(DEVICE)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_valid_bleu = -1
    valid_bleu_scores = []
    valid_losses = []
    train_losses = []

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(translations_dir, exist_ok=True)

    if not GENERATE_ONLY:
        """Train"""
        print('🌋 start training..')

        for epoch in range(1, max_epochs + 1):

            start_time = time.time()

            train_loss = train(
                model=model,
                iterator=train_iter,
                optimizer=optimizer,
                curr_epoch=epoch,
                max_epochs=max_epochs,
            )
            valid_dict = evaluate(
                model=model,
                iterator=dev_iter,
                beam_size=beam_size,
                references=dev_references,
            )

            valid_bleu_scores.append(valid_dict['bleu_score'])
            valid_losses.append(valid_dict['loss'])
            train_losses.append(train_loss)

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            print(
                f'Epoch: {epoch:04} | Time: {epoch_mins}m_{epoch_secs}s | '
                f'train_loss: {train_loss:.3f} | '
                f'val_loss: {valid_dict["loss"]:.3f} | '
                f'val_bleu: {valid_dict["bleu_score"]:.1f}'
            )

            if valid_dict['bleu_score'] > best_valid_bleu:
                print('\t--Found new best val BLEU')
                best_valid_bleu = valid_dict['bleu_score']
                torch.save(
                    {
                        'model_state_dict': model.state_dict(),
                        'optim_dict': optimizer.state_dict(),
                    },
                    os.path.join(model_dir, 'checkpoint.pt'),
                )
                # reset patience
                patience = 0
                print(f'\tcurrent patience value: {patience}/{early_stopping_patience}')
            elif len(valid_bleu_scores) <= early_stopping_patience:
                patience += 1
                print(f'\tcurrent patience value: {patience}/{early_stopping_patience}')
            else:
                patience += 1
                print(f'\tcurrent patience value: {patience}/{early_stopping_patience}')
                if patience == early_stopping_patience:
                    print('\t--STOPPING EARLY')
                    break

    """Evaluate"""

    # load checkpoint
    print(f'load checkpoint from {model_dir}')
    #checkpoint_dict = torch.load(os.path.join(model_dir, 'checkpoint.pt'), map_location=DEVICE)
    #added for memory issue fix (after one epoch, failing during loading checkpoint)
    checkpoint_dict = torch.load(os.path.join(model_dir, 'checkpoint.pt'))

    # load model weights
    print(f'load model weights from checkpoint in {model_dir}')
    model.load_state_dict(checkpoint_dict['model_state_dict'])
    #added for memory issue fix (after one epoch, failing during loading checkpoint)
    model.to(device)

    print(f'🔥 start translating..')
    for split_name, iterator, references in zip(['dev', 'test', 'challenge'],
                                                [dev_iter, test_iter, challenge_iter],
                                                [dev_references, test_references, challenge_references]):
        result_dict = evaluate(
            model=model,
            iterator=iterator,
            beam_size=beam_size,
            references=references,
        )
        print(f'{split_name.upper()}: LOSS {result_dict["loss"]:.3f} BLEU {result_dict["bleu_score"]:.1f}')
        with open(os.path.join(translations_dir, f'gen_{split_name}.out'), 'w', encoding='utf-8') as f:
            f.writelines(list(map(lambda x: x + '\n', result_dict['translations'])))

    return {
        'model': model,
        'train_losses': train_losses,
        'dev_losses': valid_losses,
        'dev_bleu_scores': valid_bleu_scores
    }


def main():
    print(f'device: {DEVICE}')
    print('Number of trainable model parameters:', sum(p.numel() for p in MODEL.parameters() if p.requires_grad))

    # set random seed for reproducibility
    seed_everything(SEED)

    # load data
    train_df = load_data(TRAIN_PATH)
    dev_df = load_data(DEV_PATH)
    test_df = load_data(TEST_PATH)
    challenge_df = load_data(CHALLENGE_PATH)

    # create datasets
    if MODE in ['text', 'multimodal']:
        train_dataset = TranslationDataset(train_df, 'en', TRGLANG, TOKENIZER)
        dev_dataset = TranslationDataset(dev_df, 'en', TRGLANG, TOKENIZER)
        test_dataset = TranslationDataset(test_df, 'en', TRGLANG, TOKENIZER)
        challenge_dataset = TranslationDataset(challenge_df, 'en', TRGLANG, TOKENIZER)
    else:
        raise NotImplementedError

    # create data iterators
    train_iter = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=train_dataset.collate_fn, shuffle=True)
    dev_iter = DataLoader(dev_dataset, batch_size=BATCH_SIZE, collate_fn=train_dataset.collate_fn, shuffle=False)
    test_iter = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=train_dataset.collate_fn, shuffle=False)
    challenge_iter = DataLoader(challenge_dataset, batch_size=BATCH_SIZE, collate_fn=train_dataset.collate_fn,
                                shuffle=False)

    # train till convergence and evaluate
    results_dict = train_till_convergence(
        model=MODEL,
        train_iter=train_iter,
        dev_iter=dev_iter,
        test_iter=test_iter,
        challenge_iter=challenge_iter,
        dev_references=dev_df[TRGLANG].astype(str).to_list(),
        test_references=test_df[TRGLANG].astype(str).to_list(),
        challenge_references=challenge_df[TRGLANG].astype(str).to_list(),
        lr=LEARNING_RATE,
        max_epochs=MAX_EPOCHS,
        model_dir=MODEL_DIR,
        translations_dir=TRANSLATIONS_DIR,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        beam_size=BEAM_SIZE,
    )

    # write history to file
    pd.DataFrame({
        'epoch': [ep_idx + 1 for ep_idx, _ in enumerate(results_dict['train_losses'])],
        'train_loss': results_dict['train_losses'],
        'dev_loss': results_dict['dev_losses'],
        'dev_bleu_score': results_dict['dev_bleu_scores'],
    }).to_csv(HISTORY_FILE_PATH, sep='\t', encoding='utf-8', index=False)


if __name__ == "__main__":
    main()
