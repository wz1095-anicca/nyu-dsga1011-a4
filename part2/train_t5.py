import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
import wandb

from t5_utils import initialize_model, initialize_optimizer_and_scheduler, save_model, load_model_from_checkpoint, setup_wandb
from transformers import GenerationConfig
from load_data import load_t5_data
from preprocess import decode_sql
from utils import compute_metrics, save_queries_and_records, set_random_seeds

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
PAD_IDX = 0

def get_args():
    '''
    Arguments for training. You may choose to change or extend these as you see fit.
    '''
    parser = argparse.ArgumentParser(description='T5 training loop')

    # Model hyperparameters
    parser.add_argument('--finetune', action='store_true', help="Whether to finetune T5 or not")
    
    # Training hyperparameters
    parser.add_argument('--optimizer_type', type=str, default="AdamW", choices=["AdamW"],
                        help="What optimizer to use")
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=0)

    parser.add_argument('--scheduler_type', type=str, default="cosine", choices=["none", "cosine", "linear"],
                        help="Whether to use a LR scheduler and what type to use if so")
    parser.add_argument('--num_warmup_epochs', type=int, default=1,
                        help="How many epochs to warm up the learning rate for if using a scheduler")
    parser.add_argument('--max_n_epochs', type=int, default=20,
                        help="How many epochs to train the model for")
    parser.add_argument('--patience_epochs', type=int, default=3,
                        help="If validation performance stops improving, how many epochs should we wait before stopping?")

    parser.add_argument('--use_wandb', action='store_true',
                        help="If set, we will use wandb to keep track of experiments")
    parser.add_argument('--experiment_name', type=str, default='experiment',
                        help="How should we name this experiment?")

    # Data hyperparameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=16)

    # Generation hyperparameters
    parser.add_argument('--max_gen_length', type=int, default=512,
                        help="Max number of generated tokens (longest train/dev query is 511 tokens with </s>)")
    parser.add_argument('--num_beams', type=int, default=1,
                        help="1 = greedy decoding; >1 = beam search")

    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")

    # Default runs train and evaluate on dev only. --test_only skips training, loads the best
    # checkpoint of --experiment_name, re-checks it on dev and writes the test submission files.
    parser.add_argument('--test_only', action='store_true',
                        help="Skip training; generate test predictions from an existing best checkpoint")

    args = parser.parse_args()
    return args

def train(args, model, train_loader, dev_loader, optimizer, scheduler):
    best_f1 = -1
    epochs_since_improvement = 0

    model_type = 'ft' if args.finetune else 'scr'
    checkpoint_dir = os.path.join('checkpoints', f'{model_type}_experiments', args.experiment_name)
    gt_sql_path = os.path.join(f'data/dev.sql')
    gt_record_path = os.path.join(f'records/ground_truth_dev.pkl')
    model_sql_path = os.path.join(f'results/t5_{model_type}_{args.experiment_name}_dev.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{args.experiment_name}_dev.pkl')
    for epoch in range(args.max_n_epochs):
        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        print(f"Epoch {epoch}: Average train loss was {tr_loss}")

        eval_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(args, model, dev_loader,
                                                                         gt_sql_path, model_sql_path,
                                                                         gt_record_path, model_record_path)
        print(f"Epoch {epoch}: Dev loss: {eval_loss}, Record F1: {record_f1}, Record EM: {record_em}, SQL EM: {sql_em}")
        print(f"Epoch {epoch}: {error_rate*100:.2f}% of the generated outputs led to SQL errors")

        if args.use_wandb:
            result_dict = {
                'train/loss' : tr_loss,
                'dev/loss' : eval_loss,
                'dev/record_f1' : record_f1,
                'dev/record_em' : record_em,
                'dev/sql_em' : sql_em,
                'dev/error_rate' : error_rate,
            }
            wandb.log(result_dict, step=epoch)

        if record_f1 > best_f1:
            best_f1 = record_f1
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        save_model(checkpoint_dir, model, best=False)
        if epochs_since_improvement == 0:
            save_model(checkpoint_dir, model, best=True)

        if epochs_since_improvement >= args.patience_epochs:
            break

def train_epoch(args, model, train_loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss()

    for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(train_loader):
        optimizer.zero_grad()
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        decoder_input = decoder_input.to(DEVICE)
        decoder_targets = decoder_targets.to(DEVICE)

        logits = model(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
            decoder_input_ids=decoder_input,
        )['logits']

        non_pad = decoder_targets != PAD_IDX
        loss = criterion(logits[non_pad], decoder_targets[non_pad])
        loss.backward()
        optimizer.step()
        if scheduler is not None: 
            scheduler.step()

        with torch.no_grad():
            num_tokens = torch.sum(non_pad).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    return total_loss / total_tokens
        
def eval_epoch(args, model, dev_loader, gt_sql_pth, model_sql_path, gt_record_path, model_record_path):
    '''
    You must implement the evaluation loop to be using during training. We recommend keeping track
    of the model loss on the SQL queries, the metrics compute_metrics returns (save_queries_and_records should be helpful)
    and the model's syntax error rate. 

    To compute non-loss metrics, you will need to perform generation with the model. Greedy decoding or beam search
    should both provide good results. If you find that this component of evaluation takes too long with your compute,
    we found the cross-entropy loss (in the evaluation set) to be well (albeit imperfectly) correlated with F1 performance.
    '''
    model.eval()
    tokenizer = dev_loader.dataset.tokenizer
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    total_tokens = 0
    generated_queries = []

    with torch.no_grad():
        for encoder_input, encoder_mask, decoder_input, decoder_targets, initial_decoder_inputs in tqdm(dev_loader):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            decoder_input = decoder_input.to(DEVICE)
            decoder_targets = decoder_targets.to(DEVICE)

            # 1) Dev loss with teacher forcing, computed exactly like the training loss (padding ignored).
            logits = model(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                decoder_input_ids=decoder_input,
            )['logits']
            non_pad = decoder_targets != PAD_IDX
            loss = criterion(logits[non_pad], decoder_targets[non_pad])
            num_tokens = torch.sum(non_pad).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

            # 2) Free-running generation, which is what the metrics are computed on.
            generated_queries += generate_queries(args, model, tokenizer, encoder_input, encoder_mask,
                                                  initial_decoder_inputs)

    # 3) Execute the generated queries, save queries + records, and compare against the ground truth.
    save_queries_and_records(generated_queries, model_sql_path, model_record_path)
    sql_em, record_em, record_f1, model_error_msgs = compute_metrics(gt_sql_pth, model_sql_path,
                                                                     gt_record_path, model_record_path)
    # A non-empty error message means the query failed to execute (syntax error, bad column, timeout).
    error_rate = sum(1 for msg in model_error_msgs if msg) / len(model_error_msgs)

    return total_loss / total_tokens, record_f1, record_em, sql_em, error_rate

def generate_queries(args, model, tokenizer, encoder_input, encoder_mask, initial_decoder_inputs):
    '''
    Generate SQL strings for one batch. Decoding starts from the same decoder start token used in
    training (initial_decoder_inputs) and stops at </s> or after max_gen_length tokens.
    '''
    outputs = model.generate(
        input_ids=encoder_input,
        attention_mask=encoder_mask,
        decoder_input_ids=initial_decoder_inputs.to(DEVICE),
        max_new_tokens=args.max_gen_length,
        num_beams=args.num_beams,
        early_stopping=args.num_beams > 1,
    )
    # decode_sql drops special tokens, keeps original spacing and maps the "<" placeholder back.
    return [decode_sql(tokenizer, ids) for ids in outputs]

def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    '''
    You must implement inference to compute your model's generated SQL queries and its associated 
    database records. Implementation should be very similar to eval_epoch.
    '''
    model.eval()
    tokenizer = test_loader.dataset.tokenizer
    generated_queries = []

    # Same generation as eval_epoch; the test set has no targets, so no loss or metrics.
    # test_loader does not shuffle, so outputs stay in the order of data/test.nl.
    with torch.no_grad():
        for encoder_input, encoder_mask, initial_decoder_inputs in tqdm(test_loader):
            generated_queries += generate_queries(args, model, tokenizer, encoder_input.to(DEVICE),
                                                  encoder_mask.to(DEVICE), initial_decoder_inputs)

    save_queries_and_records(generated_queries, model_sql_path, model_record_path)

def main():
    # Get key arguments
    args = get_args()
    # Seed Python, NumPy and PyTorch (CPU + CUDA) so runs are reproducible.
    set_random_seeds(args.seed)
    if args.use_wandb:
        # Recommended: Using wandb (or tensorboard) for result logging can make experimentation easier
        setup_wandb(args)

    # Load the data and the model
    train_loader, dev_loader, test_loader = load_t5_data(args.batch_size, args.test_batch_size)

    # Train (skipped with --test_only, which reuses the checkpoint saved by an earlier run)
    if not args.test_only:
        model = initialize_model(args)
        optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))
        train(args, model, train_loader, dev_loader, optimizer, scheduler)

    # Evaluate the best checkpoint (by dev Record F1)
    model = load_model_from_checkpoint(args, best=True)
    model.eval()
    
    # Dev set
    experiment_name = args.experiment_name
    model_type = 'ft' if args.finetune else 'scr'
    gt_sql_path = os.path.join(f'data/dev.sql')
    gt_record_path = os.path.join(f'records/ground_truth_dev.pkl')
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_dev.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_dev.pkl')
    dev_loss, dev_record_f1, dev_record_em, dev_sql_em, dev_error_rate = eval_epoch(args, model, dev_loader,
                                                                                    gt_sql_path, model_sql_path,
                                                                                    gt_record_path, model_record_path)
    print(f"Dev set results: Loss: {dev_loss}, Record F1: {dev_record_f1}, Record EM: {dev_record_em}, SQL EM: {dev_sql_em}")
    print(f"Dev set results: {dev_error_rate*100:.2f}% of the generated outputs led to SQL errors")

    # Test set: only for the final, chosen checkpoint, so model selection never looks at the test set.
    # Files are written directly under the submission names required by the README
    # (fine-tuned: t5_ft_experiment_test, from scratch / extra credit: t5_ft_experiment_ec_test).
    if args.test_only:
        submission_name = 't5_ft_experiment_test' if args.finetune else 't5_ft_experiment_ec_test'
        model_sql_path = os.path.join(f'results/{submission_name}.sql')
        model_record_path = os.path.join(f'records/{submission_name}.pkl')
        test_inference(args, model, test_loader, model_sql_path, model_record_path)
        print(f"Test predictions from experiment '{experiment_name}' saved to {model_sql_path} and {model_record_path}")

if __name__ == "__main__":
    main()
