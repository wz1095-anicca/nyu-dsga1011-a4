import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

from preprocess import preprocess_nl, preprocess_sql

# T5's pad token id. train_t5.py ignores target positions equal to PAD_IDX in the loss.
PAD_IDX = 0
# First decoder input token. We use T5's own decoder start token (the pad token, id 0): it is what
# T5 was pre-trained with and what model.generate() uses by default, so training and generation
# start the decoder the same way without extra configuration.
DECODER_START_IDX = 0

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        # The test split has no SQL file, so it only provides encoder inputs.
        self.split = split
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        self.encoder_ids, self.target_ids = self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        # Encoder inputs: pre-processed natural language, tokenized with </s> appended.
        # No truncation: the longest input is 59 tokens.
        nl_lines = load_lines(os.path.join(data_folder, f'{split}.nl'))
        encoder_ids = [
            torch.tensor(tokenizer(preprocess_nl(line)).input_ids, dtype=torch.long) for line in nl_lines
        ]

        if split == 'test':
            return encoder_ids, None

        # Decoder targets: pre-processed SQL ("<" -> placeholder), tokenized with </s> appended so the
        # model learns where to stop. No truncation: the longest query is 511 tokens including </s>,
        # and cutting a query would make the target an invalid SQL statement.
        sql_lines = load_lines(os.path.join(data_folder, f'{split}.sql'))
        assert len(sql_lines) == len(nl_lines), f'{split}: {len(nl_lines)} NL vs {len(sql_lines)} SQL lines'
        target_ids = [
            torch.tensor(tokenizer(preprocess_sql(line)).input_ids, dtype=torch.long) for line in sql_lines
        ]
        return encoder_ids, target_ids

    def __len__(self):
        return len(self.encoder_ids)

    def __getitem__(self, idx):
        # Test examples are (encoder_ids,); train/dev examples are (encoder_ids, target_ids).
        if self.target_ids is None:
            return (self.encoder_ids[idx],)
        return self.encoder_ids[idx], self.target_ids[idx]

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_seqs = [example[0] for example in batch]
    target_seqs = [example[1] for example in batch]

    # Dynamic padding: pad each batch only to its own longest sequence.
    encoder_ids = pad_sequence(encoder_seqs, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    decoder_targets = pad_sequence(target_seqs, batch_first=True, padding_value=PAD_IDX)

    # Teacher forcing: the decoder input is the target shifted right by one position, starting with
    # the decoder start token, so at step t the decoder sees target tokens < t and predicts token t.
    batch_size = decoder_targets.size(0)
    start = torch.full((batch_size, 1), DECODER_START_IDX, dtype=torch.long)
    decoder_inputs = torch.cat([start, decoder_targets[:, :-1]], dim=1)

    # Used to start generation during evaluation.
    initial_decoder_inputs = start.clone()

    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # Same encoder padding as normal_collate_fn; there are no targets on the test set.
    encoder_seqs = [example[0] for example in batch]
    encoder_ids = pad_sequence(encoder_seqs, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    initial_decoder_inputs = torch.full((len(batch), 1), DECODER_START_IDX, dtype=torch.long)
    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    # TODO
    return train_x, train_y, dev_x, dev_y, test_x