
import torch
import torch.nn.functional as F

# change it with respect to the original model
from .llama.config import LlamaConfig
from .llama.llama import load_pretrained
from .llama.sp_tokenizer import Tokenizer

def load_backbone(config):
	"""Load the pretrained backbone to adapt.

	Two shapes of checkpoint occur in Part 5. `stories42M.pt` carries its own
	`model_args`, so `load_pretrained` can rebuild it unaided. A model you
	trained in Part 3 is a bare state dict whose architecture lives in the YAML
	config you trained with, so Q5.3 passes that config through as
	`backbone_config`.
	"""
	backbone_config = getattr(config, "backbone_config", None)
	if backbone_config is None:
		return load_pretrained(config.pretrained_model_path)

	from omegaconf import OmegaConf
	from .llama import Llama
	from .llama.sp_tokenizer import Tokenizer as _Tok

	cfg = OmegaConf.load(backbone_config)
	model = Llama(LlamaConfig(vocab_size=_Tok().n_words, **cfg.model_config))
	state = torch.load(config.pretrained_model_path, map_location="cpu", weights_only=False)
	model.load_state_dict(state.get("model", state))
	return model


class LlamaZeroShotClassifier(torch.nn.Module):
	def __init__(self, config: LlamaConfig, tokenizer: Tokenizer, label_names: list[str]):
		super(LlamaZeroShotClassifier, self).__init__()
		self.num_labels = config.num_labels
		self.llama = load_backbone(config)
		# Zero-shot classification does not require updating llama paramters.
		for param in self.llama.parameters():
			param.requires_grad = False
		assert len(label_names) == self.num_labels
		self.tokenizer = tokenizer
		self.label_name_ids = [tokenizer.encode(label, bos=False, eos=False) for label in label_names]
		# Scoring a label by summing the log-probs of its token ids at a single
		# position is only valid when each label is one token. Fail loudly rather
		# than silently mis-scoring if someone edits the label mapping.
		multi = [(n, ids) for n, ids in zip(label_names, self.label_name_ids) if len(ids) != 1]
		assert not multi, (
			f"Label names must be single tokens under this tokenizer; got {multi}. "
			f"Multi-token labels need proper sequence scoring, not a sum at one position."
		)

	def forward(self, input_ids):
		# compute the completion probability of each label string
		#
		# Batches are RIGHT-padded to the longest sequence, so the final position is a
		# PAD token for every sequence that isn't the longest one. Scoring there asks
		# the model what follows the padding rather than what follows the prompt.
		# Attention is causal, so real-token
		# representations are unaffected by trailing pads — we just have to read the
		# last REAL position for each example.
		_, hidden_states = self.llama(input_ids)          # (B, S, dim), already normed
		lengths = (input_ids != self.tokenizer.pad_id).sum(dim=1)   # (B,)
		last_real = (lengths - 1).clamp(min=0)
		h_last = hidden_states[torch.arange(hidden_states.size(0), device=hidden_states.device), last_real]
		logits = self.llama.output(h_last)                # (B, vocab)

		log_probabilities = F.log_softmax(logits, dim=-1)
		label_probabilities = torch.zeros((log_probabilities.shape[0], self.num_labels), device=log_probabilities.device)
		for i, label_token_ids in enumerate(self.label_name_ids):
			label_probabilities[:, i] = log_probabilities[:, label_token_ids].sum(dim=-1)
		return label_probabilities

class LlamaEmbeddingClassifier(torch.nn.Module):
	def __init__(self, config):
		super(LlamaEmbeddingClassifier, self).__init__()
		self.num_labels = config.num_labels
		self.llama = load_backbone(config)
		# If we use pretrain mode, we freeze Llama parameters.
		for param in self.llama.parameters():
			if config.option == 'pretrain':
				param.requires_grad = False
			elif config.option == 'finetune':
				param.requires_grad = True

		self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
		self.classifier_head = torch.nn.Linear(self.llama.config.dim, self.num_labels)
		# Needed to locate the last real token: batches are right-padded, so
		# hidden_states[:, -1] is a PAD position for most examples.
		self.pad_id = getattr(config, "pad_id", 3)

	def forward(self, input_ids):
		'''
		1) Find the hidden state after the final NON-PADDING token of the input sequence.
		   Batches are right-padded to the longest sequence, so hidden_states[:, -1] is a
		   pad position for every sequence that isn't the longest in its batch. Attention
		   is causal, so real-token representations are unaffected by trailing pads — you
		   just have to index the right position.
		2) Apply dropout (self.dropout) to the hidden state at training time to mitigate
		   overfitting.
		2) Pass this through the classifier head (self.classifier_head), which will return
		   logits (unnormalized probabilities) over all classes.
		3) Take the log-softmax of the logits and return log-probabilities over all classes.
		'''
		# todo: pool the last non-padding hidden state, then dropout + classifier head + log-softmax
		raise NotImplementedError

# ============================================================================
# Part 5 driver
#
# Notes:
#   * the prompt deliberately does not end in a trailing space (see the module
#     docstring on classifier scoring), and
#   * a backbone config can be supplied so Q5.3 can finetune a model you
#     trained in Part 3, not only the provided stories42M checkpoint.
# ============================================================================

import argparse
import json
import os
import random
from types import SimpleNamespace

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .optim import AdamW


def seed_everything(seed=11711):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True


class SentimentDataset(Dataset):
	def __init__(self, dataset, tokenizer, eos=False):
		self.dataset = dataset
		self.tokenizer = tokenizer
		self.eos = eos

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx):
		return self.dataset[idx]

	def collate_fn(self, all_data):
		sents = [x[0] for x in all_data]
		labels = [x[1] for x in all_data]
		encoding = [self.tokenizer.encode(s, bos=True, eos=self.eos) for s in sents]
		width = max(len(e) for e in encoding)
		# right-padded; the classifiers locate the last real token themselves
		padded = [e + [self.tokenizer.pad_id] * (width - len(e)) for e in encoding]
		return {"token_ids": torch.LongTensor(padded),
		        "labels": torch.LongTensor(labels),
		        "sents": sents}


def create_data(filename, tokenizer, flag="train", eos=True, prompt_suffix=None):
	num_labels, data = {}, []
	with open(filename) as fp:
		for line in fp:
			label, org_sent = line.split(" ||| ")
			sent = org_sent.strip()
			if prompt_suffix is not None:
				sent = f"{sent} {prompt_suffix}"
			label = int(label.strip())
			num_labels.setdefault(label, len(num_labels))
			data.append((sent, label))
	print(f"loaded {len(data)} examples from {filename}")
	return (data, len(num_labels)) if flag == "train" else data


@torch.no_grad()
def model_eval(dataloader, model, device):
	model.eval()
	y_true, y_pred, sents = [], [], []
	for batch in tqdm(dataloader, desc="eval", leave=False):
		logits = model(batch["token_ids"].to(device)).detach().cpu().numpy()
		y_pred.extend(np.argmax(logits, axis=1).flatten())
		y_true.extend(batch["labels"].flatten().tolist())
		sents.extend(batch["sents"])
	return (accuracy_score(y_true, y_pred),
	        f1_score(y_true, y_pred, average="macro"), y_pred, y_true, sents)


def write_predictions(split, outfile, acc, preds, sents, gold=None):
	# CFIMDB's test file ships placeholder labels (every label is the same
	# class), so accuracy against it is meaningless -- and actively misleading:
	# a constant predictor "scores" ~0.93 on it while a genuinely working
	# classifier sits near 0.5. Refuse to print a number rather than invite
	# that comparison. Report dev accuracy only; predictions are still written.
	os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)
	if gold is not None and len(set(gold)) <= 1:
		print(f"{split} acc :: n/a (your copy has placeholder labels; "
		      f"predictions written for staff scoring, report dev only)")
	else:
		print(f"{split} acc :: {acc:.3f}")
	with open(outfile, "w") as f:
		for s, p in zip(sents, preds):
			f.write(f"{p} ||| {s}\n")


def _base_config(args, num_labels, tokenizer):
	return SimpleNamespace(
		hidden_dropout_prob=args.hidden_dropout_prob,
		pretrained_model_path=args.pretrained_model_path,
		backbone_config=args.backbone_config,
		num_labels=num_labels, data_dir=".", option=args.option,
		pad_id=tokenizer.pad_id)


def run_prompting(args, tokenizer, device):
	label_names = json.load(open(args.label_names))
	_, num_labels = create_data(args.train, tokenizer, "train")

	if len(label_names) == 2:
		label_str = " or ".join(label_names)
	else:
		label_str = ", ".join(label_names[:-1]) + ", or " + label_names[-1]
	# No trailing space: it would tokenize to a dangling U+2581 while every
	# label already carries its own leading-space marker.
	suffix = f"Is this movie {label_str}? This movie is"

	config = _base_config(args, num_labels, tokenizer)
	config.label_names = label_names
	model = LlamaZeroShotClassifier(config, tokenizer, label_names).to(device)

	for split, path, out in [("dev", args.dev, args.dev_out), ("test", args.test, args.test_out)]:
		data = create_data(path, tokenizer, "valid", eos=False, prompt_suffix=suffix)
		ds = SentimentDataset(data, tokenizer, eos=False)
		dl = DataLoader(ds, shuffle=False, batch_size=args.batch_size, collate_fn=ds.collate_fn)
		acc, f1, preds, gold, sents = model_eval(dl, model, device)
		write_predictions(split, out, acc, preds, sents, gold)


def run_finetuning(args, tokenizer, device):
	train_data, num_labels = create_data(args.train, tokenizer, "train")
	dev_data = create_data(args.dev, tokenizer, "valid")

	train_ds = SentimentDataset(train_data, tokenizer)
	dev_ds = SentimentDataset(dev_data, tokenizer)
	train_dl = DataLoader(train_ds, shuffle=True, batch_size=args.batch_size,
	                      collate_fn=train_ds.collate_fn)
	dev_dl = DataLoader(dev_ds, shuffle=False, batch_size=args.batch_size,
	                    collate_fn=dev_ds.collate_fn)

	config = _base_config(args, num_labels, tokenizer)
	model = LlamaEmbeddingClassifier(config).to(device)
	optimizer = AdamW(model.parameters(), lr=args.lr)

	best_dev_acc = 0.0
	for epoch in range(args.epochs):
		model.train()
		total, n = 0.0, 0
		for batch in tqdm(train_dl, desc=f"train-{epoch}", leave=False):
			ids = batch["token_ids"].to(device)
			labels = batch["labels"].to(device)
			optimizer.zero_grad()
			loss = F.nll_loss(model(ids), labels.view(-1), reduction="mean")
			loss.backward()
			optimizer.step()
			total += loss.item(); n += 1
		dev_acc, dev_f1, preds, dev_gold, sents = model_eval(dev_dl, model, device)
		print(f"epoch {epoch}: train loss {total/max(n,1):.3f}, dev acc {dev_acc:.3f}")
		if dev_acc > best_dev_acc:
			best_dev_acc = dev_acc
			torch.save({"model": model.state_dict(), "config": config}, args.filepath)
			write_predictions("dev", args.dev_out, dev_acc, preds, sents, dev_gold)

	print(f"best dev accuracy: {best_dev_acc:.3f}")

	test_data = create_data(args.test, tokenizer, "test")
	test_ds = SentimentDataset(test_data, tokenizer)
	test_dl = DataLoader(test_ds, shuffle=False, batch_size=args.batch_size,
	                     collate_fn=test_ds.collate_fn)
	acc, _, preds, gold, sents = model_eval(test_dl, model, device)
	write_predictions("test", args.test_out, acc, preds, sents, gold)


def get_args():
	p = argparse.ArgumentParser(description="Part 5 — adapt a pretrained Llama.")
	p.add_argument("--option", choices=("prompt", "finetune"), required=True,
	               help="prompt: backbone frozen, no training; finetune: train a head")
	p.add_argument("--pretrained-model-path", default="data/stories42M.pt")
	p.add_argument("--backbone-config", default=None,
	               help="YAML your backbone was trained with — needed for Q5.3, "
	                    "when adapting a model you trained in Part 3")
	p.add_argument("--train", default="data/sst-train.txt")
	p.add_argument("--dev", default="data/sst-dev.txt")
	p.add_argument("--test", default="data/sst-test.txt")
	p.add_argument("--label-names", default="data/sst-label-mapping.json")
	p.add_argument("--dev-out", default=None,
	               help="default: outputs/preds/<split>-<option>-output.txt")
	p.add_argument("--test-out", default=None,
	               help="default: outputs/preds/<split>-<option>-output.txt")
	p.add_argument("--epochs", type=int, default=5)
	p.add_argument("--batch-size", type=int, default=32)
	p.add_argument("--lr", type=float, default=2e-5)
	p.add_argument("--hidden-dropout-prob", type=float, default=0.3)
	p.add_argument("--seed", type=int, default=1337)
	p.add_argument("--use-gpu", action="store_true")
	p.add_argument("--output-dir", default="outputs/adapt",
	               help="where to write the finetuned checkpoint")
	args = p.parse_args()
	# Write into an output directory rather than the current one, so large
	# finetuned checkpoints (~190MB) do not litter the repo root, where they
	# would be easy to mistake for something you are meant to submit.
	os.makedirs(args.output_dir, exist_ok=True)
	args.filepath = os.path.join(
		args.output_dir, f"{args.option}-{args.epochs}-{args.lr}.pt")
	# Prediction files are part of the code submission (zip-submission.sh picks
	# up outputs/preds/), so give each dataset x option pair its own name. A
	# Q5.3 run (your own backbone) gets an -own suffix so it cannot overwrite
	# the Q5.2 predictions from the provided checkpoint.
	tag = "-own" if args.backbone_config else ""
	for attr, split in (("dev_out", args.dev), ("test_out", args.test)):
		if getattr(args, attr) is None:
			stem = os.path.splitext(os.path.basename(split))[0]
			setattr(args, attr, os.path.join(
				"outputs", "preds", f"{stem}-{args.option}{tag}-output.txt"))
	return args


def main():
	args = get_args()
	seed_everything(args.seed)
	device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
	tokenizer = Tokenizer()
	if args.option == "prompt":
		run_prompting(args, tokenizer, device)
	else:
		run_finetuning(args, tokenizer, device)


if __name__ == "__main__":
	main()
