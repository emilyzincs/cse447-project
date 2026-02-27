#!/usr/bin/env python
import os
import sys
import random
import pickle
import traceback
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from itertools import islice

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset

from utils import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CharTransformer(nn.Module):
    """
    Character level transformer model.

    Layers: character embedding, positional embedding, transformer encoder, output linear layer.
    """
    def __init__(self, vocab_size, pad_idx, d_model=128, nhead=4, num_layers=2, max_len=512):
        super().__init__()

        # maps each character index to a d_model-dim vector
        self.embed = nn.Embedding(vocab_size, d_model)
        # maps positions, allows for learning positional info
        self.pos_embed = nn.Embedding(max_len, d_model)
        # transformer encoder layers, with self-attention and feedforward
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        # output, maps back to vocab size for prediction
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.pad_idx = pad_idx

        # Precompute mask for attention
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(max_len, max_len), diagonal=1)
        )

    def forward(self, x):
      B, T = x.shape
      positions = torch.arange(T, device=x.device).unsqueeze(0)

      x_embed = self.embed(x) + self.pos_embed(positions)

      causal_mask = self.mask[:T, :T].bool()

      padding_mask = (x == self.pad_idx)

      out = self.transformer(
          x_embed,
          mask=causal_mask,
          src_key_padding_mask=padding_mask
      )

      logits = self.fc_out(out)
      return logits


class TransformerModel:
    """
    Model wrapper class.
    """

    def __init__(self):
        self.PREFIX_CHAR = '|'
        self.PAD_CHAR = '<PAD>' # to ensure we can process diff input lengths in batches
        self.max_len = 256 # max sequence length for training/prediction

        self.char2idx = {}
        self.idx2char = {}
        self.pad_idx = None

        self.model = None

    @classmethod
    def load_training_data(cls, languages, max_docs_per_lang, streaming):
        """
        Load training data.
        """
        data = []
        dataset_name = "Cohere/wikipedia-2023-11-embed-multilingual-v3"

        for lang in languages:
            docs_loaded = 0
            print(f"Loading {lang}...")

            docs_stream = load_dataset(
                dataset_name,
                lang,
                split="train",
                streaming=streaming
            )

            for doc in islice(docs_stream, max_docs_per_lang):
                text = doc.get("text", "")
                if text:
                    data.append(text)
                    docs_loaded += 1
                if docs_loaded >= max_docs_per_lang:
                    break

        return data

    
    def build_vocab(self, data):
        """
        Build character vocabulary from training data, plus special chars.
        """
        charset = set()
        for line in data:
            for ch in line:
                charset.add(ch)
        charset.add(self.PREFIX_CHAR)
        charset.add(self.PAD_CHAR)
        charset = sorted(list(charset))
        self.char2idx = {c: i for i, c in enumerate(charset)}
        self.idx2char = {i: c for c, i in self.char2idx.items()}
        self.pad_idx = self.char2idx[self.PAD_CHAR]
        print("Vocab size:", len(self.char2idx))

    def encode(self, text):
        # Only encode characters in vocab; skip unknowns
        indices = []
        for c in text:
            if c in self.char2idx:
                indices.append(self.char2idx[c])
            else:
                # Optionally warn or skip
                pass
        return indices

    def decode(self, indices):
        return ''.join([self.idx2char[i] for i in indices])

   
    def run_train(self, data, work_dir):

        print("Building vocab...")
        self.build_vocab(data)

        vocab_size = len(self.char2idx)

        self.model = CharTransformer(
          vocab_size,
          self.pad_idx,
          d_model=getattr(self, 'd_model', 128),
          nhead=getattr(self, 'nhead', 4),
          num_layers=getattr(self, 'num_layers', 2),
          max_len=getattr(self, 'max_len', 256)
      ).to(DEVICE)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)

        batch_size = getattr(self, 'batch_size', 32)
        epochs = getattr(self, 'epochs', 1)

        print("Training...")

        losses = []
        top1_accs = []
        top3_accs = []
        val_losses_all = []
        val_top1s_all = []
        val_top3s_all = []

        # Prepare dataset, chunk each document into many max_len segments
        encoded_data = []
        for line in data:
            text = (self.PREFIX_CHAR * 3) + line
            encoded = self.encode(text)
            for i in range(0, len(encoded) - 1, self.max_len):
                chunk = encoded[i:i + self.max_len]
                if len(chunk) > 1:
                    encoded_data.append(chunk)

        # Split into train/val
        split_idx = int(0.9 * len(encoded_data))
        train_data = encoded_data[:split_idx]
        val_data = encoded_data[split_idx:]

        def eval_batches(data_batches):
            self.model.eval()
            val_losses = []
            val_top1s = []
            val_top3s = []
            with torch.no_grad():
                for batch_start in range(0, len(data_batches), batch_size):
                    batch = data_batches[batch_start:batch_start+batch_size]
                    if not batch:
                        continue
                    x_list, y_list = [], []
                    for encoded in batch:
                        x_list.append(encoded[:-1])
                        y_list.append(encoded[1:])
                    max_seq = max(len(x) for x in x_list)
                    B = len(x_list)
                    x_tensor = torch.full((B, max_seq), self.pad_idx, dtype=torch.long).to(DEVICE)
                    y_tensor = torch.full((B, max_seq), self.pad_idx, dtype=torch.long).to(DEVICE)
                    for i, (x, y) in enumerate(zip(x_list, y_list)):
                        x_tensor[i, :len(x)] = torch.tensor(x)
                        y_tensor[i, :len(y)] = torch.tensor(y)
                    logits = self.model(x_tensor)
                    loss = F.cross_entropy(
                        logits.view(-1, vocab_size),
                        y_tensor.view(-1),
                        ignore_index=self.pad_idx
                    )
                    preds_top1 = logits.argmax(dim=-1)
                    mask_valid = y_tensor != self.pad_idx
                    correct = (preds_top1 == y_tensor) & mask_valid
                    acc_top1 = correct.sum().float() / mask_valid.sum().float() if mask_valid.sum() > 0 else 0.0
                    topk = torch.topk(logits, 3, dim=-1).indices
                    target_exp = y_tensor.unsqueeze(-1).expand_as(topk)
                    in_top3 = ((topk == target_exp) & mask_valid.unsqueeze(-1)).any(dim=-1).float().mean().item()
                    val_losses.append(loss.item())
                    val_top1s.append(acc_top1.item())
                    val_top3s.append(in_top3 if isinstance(in_top3, float) else in_top3.item())
            self.model.train()
            return val_losses, val_top1s, val_top3s

        for epoch in range(epochs):
            random.shuffle(train_data)
            num_batches = (len(train_data) + batch_size - 1) // batch_size
            pbar = tqdm(range(0, len(train_data), batch_size), desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
            for batch_start in pbar:
                batch = train_data[batch_start:batch_start+batch_size]
                x_list, y_list = [], []
                for encoded in batch:
                    x_list.append(encoded[:-1])
                    y_list.append(encoded[1:])
                max_seq = max(len(x) for x in x_list)
                B = len(x_list)
                x_tensor = torch.full((B, max_seq), self.pad_idx, dtype=torch.long).to(DEVICE)
                y_tensor = torch.full((B, max_seq), self.pad_idx, dtype=torch.long).to(DEVICE)
                for i, (x, y) in enumerate(zip(x_list, y_list)):
                    x_tensor[i, :len(x)] = torch.tensor(x)
                    y_tensor[i, :len(y)] = torch.tensor(y)

                logits = self.model(x_tensor)
                loss = F.cross_entropy(
                    logits.view(-1, vocab_size),
                    y_tensor.view(-1),
                    ignore_index=self.pad_idx
                )

                with torch.no_grad():
                    preds_top1 = logits.argmax(dim=-1)
                    mask_valid = y_tensor != self.pad_idx
                    correct = (preds_top1 == y_tensor) & mask_valid
                    acc_top1 = correct.sum().float() / mask_valid.sum().float() if mask_valid.sum() > 0 else 0.0
                    topk = torch.topk(logits, 3, dim=-1).indices
                    target_exp = y_tensor.unsqueeze(-1).expand_as(topk)
                    in_top3 = ((topk == target_exp) & mask_valid.unsqueeze(-1)).any(dim=-1).float().mean().item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                top1_accs.append(acc_top1.item())
                top3_accs.append(in_top3 if isinstance(in_top3, float) else in_top3.item())

                if (batch_start // batch_size + 1) % 10 == 0:
                    pbar.set_postfix({"loss": f"{loss.item():.4f}", "top1": f"{acc_top1:.4f}", "top3": f"{in_top3:.4f}"})

            # Validation after each epoch
            val_losses, val_top1s, val_top3s = eval_batches(val_data)
            val_losses_all.extend(val_losses)
            val_top1s_all.extend(val_top1s)
            val_top3s_all.extend(val_top3s)

        print("Training complete")

        # Summary metrics
        if len(losses) > 0:
            final_loss = losses[-1]
            mean_loss = float(np.mean(losses))
            final_top1 = top1_accs[-1]
            mean_top1 = float(np.mean(top1_accs))
            final_top3 = top3_accs[-1]
            mean_top3 = float(np.mean(top3_accs))

            print("Training summary metrics:")
            print(f"  Final loss: {final_loss:.4f}")
            print(f"  Mean loss: {mean_loss:.4f}")
            print(f"  Final top-1 accuracy: {final_top1:.4f}")
            print(f"  Mean top-1 accuracy: {mean_top1:.4f}")
            print(f"  Final top-3 accuracy: {final_top3:.4f}")
            print(f"  Mean top-3 accuracy: {mean_top3:.4f}")

            # Plots
            try:
                os.makedirs(work_dir, exist_ok=True)

                plt.figure()
                plt.plot(losses, label='train loss')
                if val_losses_all:
                    plt.plot(np.linspace(0, len(losses), len(val_losses_all)), val_losses_all, label='val loss')
                plt.xlabel('step')
                plt.ylabel('loss')
                plt.title('Loss (Train/Val)')
                plt.grid(True)
                plt.legend()
                loss_path = os.path.join(work_dir, 'loss.png')
                plt.savefig(loss_path)
                plt.close()

                plt.figure()
                plt.plot(top1_accs, label='train top1')
                plt.plot(top3_accs, label='train top3')
                if val_top1s_all:
                    plt.plot(np.linspace(0, len(top1_accs), len(val_top1s_all)), val_top1s_all, label='val top1')
                if val_top3s_all:
                    plt.plot(np.linspace(0, len(top3_accs), len(val_top3s_all)), val_top3s_all, label='val top3')
                plt.xlabel('step')
                plt.ylabel('accuracy')
                plt.title('Accuracy (Train/Val)')
                plt.grid(True)
                plt.legend()
                acc_path = os.path.join(work_dir, 'accuracy.png')
                plt.savefig(acc_path)
                plt.close()

                print(f"Saved loss plot to: {loss_path}")
                print(f"Saved accuracy plot to: {acc_path}")
            except Exception as e:
                print("Failed to save plots:", e)

        return {
            'losses': losses,
            'top1_accs': top1_accs,
            'top3_accs': top3_accs,
        }

    @classmethod
    def load_test_data(cls, fname):
        data = []
        with open(fname, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                data.append(line)
        return data


    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, "w", encoding="utf-8") as f:
            for p in preds:
                f.write(f"{p}\n")

    def run_pred(self, data):

        preds = []

        for line in data:
            text = (self.PREFIX_CHAR * 3) + line
            encoded = self.encode(text)

            encoded = encoded[-self.max_len:]

            x = torch.tensor(encoded).unsqueeze(0).to(DEVICE)

            try:
                with torch.no_grad():
                    logits = self.model(x)

                probs = F.softmax(logits[0, -1], dim=-1)
                top3 = torch.topk(probs, 3).indices.tolist()
                chars = [self.idx2char[idx] for idx in top3]

            except Exception:
                chars = random.sample(list(self.char2idx.keys()), 3)

            preds.append(''.join(chars))

        return preds

    def save(self, work_dir):
        print("Saving model to:", os.path.abspath(os.path.join(work_dir, "model.checkpoint")))
        torch.save({
            "model_state": self.model.state_dict(),
            "char2idx": self.char2idx,
            "idx2char": self.idx2char,
            "d_model": getattr(self, 'd_model', 128),
            "nhead": getattr(self, 'nhead', 4),
            "num_layers": getattr(self, 'num_layers', 2),
            "max_len": getattr(self, 'max_len', 256)
        }, os.path.join(work_dir, "model.checkpoint"))

    @classmethod
    def load(cls, work_dir):
        checkpoint = torch.load(os.path.join(work_dir, "model.checkpoint"), map_location=DEVICE)
        model = cls()
        model.char2idx = checkpoint["char2idx"]
        model.idx2char = checkpoint["idx2char"]
        vocab_size = len(model.char2idx)
        d_model = checkpoint.get("d_model", 128)
        nhead = checkpoint.get("nhead", 4)
        num_layers = checkpoint.get("num_layers", 2)
        max_len = checkpoint.get("max_len", 256)
        model.model = CharTransformer(
            vocab_size,
            model.char2idx[model.PAD_CHAR],
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            max_len=max_len
        ).to(DEVICE)
        model.model.load_state_dict(checkpoint["model_state"])
        model.model.eval()
        model.d_model = d_model
        model.nhead = nhead
        model.num_layers = num_layers
        model.max_len = max_len
        return model


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', default='work', help='where to save model')
    parser.add_argument('--test_data', default='example/input.txt', help='path to test data')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size for training')
    parser.add_argument('--test_output', default='pred.txt', help='path to write predictions')

    parser.add_argument('--languages', nargs='+',
                        default=['en', 'ru', 'zh'],
                        help='languages to load from HF dataset')

    parser.add_argument('--max_docs', type=int, default=10000,
                        help='max documents per language')

    parser.add_argument('--stream', action='store_true', default=True,
                        help='stream dataset instead of downloading')

    # Transformer-specific arguments
    parser.add_argument('--d_model', type=int, default=128,
                        help='embedding dimension')
    parser.add_argument('--nhead', type=int, default=4,
                        help='number of attention heads')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of transformer layers')
    parser.add_argument('--epochs', type=int, default=1,
                        help='training epochs')
    parser.add_argument('--max_len', type=int, default=256,
                        help='maximum sequence length')

    args = parser.parse_args()
    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print(f'Making working directory {args.work_dir}')
            os.makedirs(args.work_dir)

        print('Instantiating Transformer model')
        model = TransformerModel()
        # Pass CLI args to model for training
        model.batch_size = args.batch_size
        model.epochs = args.epochs
        model.d_model = args.d_model
        model.nhead = args.nhead
        model.num_layers = args.num_layers
        model.max_len = args.max_len

        print('Loading training data')
        train_data = TransformerModel.load_training_data(
            languages=args.languages,
            max_docs_per_lang=args.max_docs,
            streaming=args.stream,
        )

        print('Training')
        model.run_train(train_data, args.work_dir)

        print('Saving model')
        model.save(args.work_dir)

    elif args.mode == 'test':
        print('Loading model')
        model = TransformerModel.load(args.work_dir)
        model.max_len = args.max_len

        print(f'Loading test data from {args.test_data}')
        test_data = TransformerModel.load_test_data(args.test_data)

        print('Making predictions')
        pred = model.run_pred(test_data)

        print(f'Writing predictions to {args.test_output}')
        model.write_pred(pred, args.test_output)