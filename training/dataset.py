"""Tokenize tapes into packed training sequences and label every token for the loss.

Alongside input_ids each sample carries a `mask` that tells the loss what a token is:
0 for the system prompt and padding (ignored), 1 for response text, and a distinct id from 2
upwards for each state transition token, so the loss can weight the transition types separately.
"""

import torch
from torch.utils.data import Dataset, ConcatDataset
from def_fsm.utils import STATE_TRANSITION_TOKENS, INTERLOCUTOR_PREFIX, system_prompt_fillin
from tqdm import tqdm
from datasets import Dataset as HFDataset
from datasets.utils.logging import disable_progress_bar, enable_progress_bar
from itertools import chain
from collections import Counter
import random
import os
import pickle
from datasets import load_from_disk
from utils import gen_maskid_to_tokenid_map


class TapeDataset(Dataset):
    def __init__(
        self,
        raw_data,
        tokenizer,
        max_length,
        system_prompt_path=None,
        verbose=True,
        interlocutor_prefix=INTERLOCUTOR_PREFIX,
        dataset_id=0,
        seed=42,
        cache_dir=None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.interlocutor_prefix = interlocutor_prefix
        self.dataset_id = dataset_id
        num_proc = 8

        if cache_dir and os.path.exists(os.path.join(cache_dir, "hf_dataset")) and os.path.exists(os.path.join(cache_dir, "metadata.pkl")):
            if verbose:
                print(f"Loading cached dataset from {cache_dir}")
            self._load_from_cache(cache_dir)
            return

        disable_progress_bar()

        # 0. Prepare the System Prompt
        if system_prompt_path:
            system_prompt = system_prompt_fillin(system_prompt_path, interlocutor_prefix)
            self.system_prompt_ids = tokenizer.encode(system_prompt)
            # System prompt mask is 0 (ignore loss)
            self.system_prompt_masks = [0] * len(self.system_prompt_ids)
        else:
            self.system_prompt_ids = []
            self.system_prompt_masks = []

        self.sys_len = len(self.system_prompt_ids)
        # reserve one slot for the EOS token (1 token)
        self.content_limit = self.max_length - self.sys_len - 1

        if self.content_limit <= 0:
            raise ValueError(f"System prompt length ({self.sys_len}) is too large for max_length ({self.max_length})!")

        # 1. Build the Mask mapping table
        self.maskid_to_tokenstr_map = {}  # Only state transition tokens
        self.tokenstr_to_maskid_map = {}  # Only state transition tokens
        sorted_tokens = sorted(list(STATE_TRANSITION_TOKENS))
        current_mask_id = 2  # start from 2; 0 is reserved for pad/ignore, 1 for plain unmasked tokens

        for token_str in sorted_tokens:
            ids = tokenizer.encode(token_str)
            self.maskid_to_tokenstr_map[current_mask_id] = token_str
            self.tokenstr_to_maskid_map[token_str] = [current_mask_id] * len(ids)
            current_mask_id += 1

        self.maskid_to_tokenid_map = gen_maskid_to_tokenid_map(self.maskid_to_tokenstr_map, tokenizer)

        # 2. Convert to HF Dataset
        rng = random.Random(seed)
        rng.shuffle(raw_data)
        hf_dataset = HFDataset.from_list(raw_data)

        # Pipeline Step 1: Basic Tokenization (no System Prompt, no EOS, no truncation)
        # this step only turns strings into IDs and keeps the original long-sequence structure
        ds_tokenized = hf_dataset.map(
            self._tokenize_raw,
            batched=False,  # per-row processing keeps the logic clearest
            remove_columns=hf_dataset.column_names,  # remove the original 'tape' text column
            desc="Step 1: Raw Tokenization",
            num_proc=num_proc,
        )

        original_lengths = ds_tokenized["length"]
        total_tokens = sum(original_lengths)
        num_conversations = len(original_lengths)

        if num_conversations > 0:
            avg_len = total_tokens / num_conversations
            max_len = max(original_lengths)
            min_len = min(original_lengths)
        else:
            avg_len = max_len = min_len = 0

        self.token_counts = self._compute_token_counts(ds_tokenized)  # Counter type

        # Pipeline Step 2: Slicing + injecting the System Prompt (Sliding Window / Chunking)
        # this step cuts long sequences into multiple short sequences, each carrying a System Prompt and EOS
        ds_sliced = ds_tokenized.map(
            self._slice_with_system_prompt,
            batched=True,  # must be True because we turn 1 row into N rows
            batch_size=1000,
            remove_columns=ds_tokenized.column_names,  # remove the long-sequence columns from Step 1
            desc="Step 2: Slicing & Sys Prompt Injection",
            num_proc=num_proc,
        )

        # Pipeline Step 3: Packing (No-Split strategy)
        # this step concatenates the short slices from Step 2 into max_length-sized batches
        self.packed_dataset = ds_sliced.map(
            self._pack_sequences,
            batched=True,
            batch_size=2000,
            remove_columns=ds_sliced.column_names,
            desc="Step 3: Packing (No-Split)",
            num_proc=num_proc,
        )

        # 4. Update Stats
        self.stats = {
            "total_conversations": num_conversations,
            "total_tokens": total_tokens,
            "total_tokens_millions": total_tokens / 1e6,
            "average_length": avg_len,
            "max_length": max_len,
            "min_length": min_len,
            "dataset_samples": len(self.packed_dataset),  # actual number of training samples (Packed)
            "tape_context_size": max_length,
            "packing_efficiency": total_tokens / (len(self.packed_dataset) * max_length) if len(self.packed_dataset) > 0 else 0,
        }

        if verbose:
            print("Tape Tokenization Report:")
            print(f"  - Total Conversations: {self.stats['total_conversations']}")
            print(f"  - Total Tokens:        {self.stats['total_tokens_millions']:.4f} million tokens")
            print(f"  - Average Length:      {self.stats['average_length']:.2f} tokens")
            print(f"  - Max Length:          {self.stats['max_length']} tokens")
            print(f"  - Min Length:          {self.stats['min_length']} tokens")
            print(f"  - Packed Dataset Samples:     {self.stats['dataset_samples']} (Block Size: {max_length})")
            print(f"  - Packing Efficiency:     {self.stats['packing_efficiency']:.2%}")

        if cache_dir:
            if verbose:
                print(f"Saving dataset cache to {cache_dir}")
            self._save_to_cache(cache_dir)

    def _tokenize_raw(self, example):
        """
        Step 1: Tokenization and Mask mapping only.
        Do not add System Prompt, do not add EOS, do not truncate.
        """
        raw_ids = []
        raw_masks = []

        for seg_str in example["tape"]:
            seg_str = seg_str.replace(INTERLOCUTOR_PREFIX, self.interlocutor_prefix)
            seg_ids = self.tokenizer.encode(seg_str)

            if seg_str in self.tokenstr_to_maskid_map:
                seg_mask = self.tokenstr_to_maskid_map[seg_str][:]
            elif not seg_str.startswith(self.interlocutor_prefix):
                seg_mask = [1] * len(seg_ids)
            else:
                seg_mask = [0] * len(seg_ids)

            raw_ids.extend(seg_ids)
            raw_masks.extend(seg_mask)

        assert len(raw_ids) == len(raw_masks), f"{len(raw_ids)=}, {len(raw_masks)=}"

        return {
            "input_ids": raw_ids,
            "mask": raw_masks,
            "length": len(raw_ids),
        }

    def _slice_with_system_prompt(self, examples):
        """
        Step 2: Slice and inject the System Prompt.
        Input: Batch of long sequences (from Step 1)
        Output: Batch of sliced sequences (flattened)
        """
        out_input_ids = []
        out_masks = []

        # examples['input_ids'] here is a list of lists
        for i in range(len(examples["input_ids"])):
            full_ids = examples["input_ids"][i]
            full_masks = examples["mask"][i]

            total_len = len(full_ids)

            # use content_limit as the stride for slicing
            # to get overlap, reduce the stride, e.g. stride = content_limit // 2
            stride = int(self.content_limit * 7 / 8)

            for start_idx in range(0, total_len, stride):
                # 1. extract the segment
                chunk_ids = full_ids[start_idx : start_idx + self.content_limit]
                chunk_masks = full_masks[start_idx : start_idx + self.content_limit]

                # 2. concatenate: [System] + [Chunk] + [EOS]
                final_ids = self.system_prompt_ids + chunk_ids
                final_masks = self.system_prompt_masks + chunk_masks

                assert self.tokenizer.eos_token_id is not None
                final_ids.append(self.tokenizer.eos_token_id)
                final_masks.append(0)

                out_input_ids.append(final_ids)
                out_masks.append(final_masks)

        return {"input_ids": out_input_ids, "mask": out_masks}

    def _pack_sequences(self, examples):
        """
        Step 3: No-split packing.
        Concatenate the short slices produced by Step 2 into sequences of length max_length.
        Ensures every complete slice generated by Step 2 is kept intact.
        """
        packed_input_ids = []
        packed_masks = []

        current_ids = []
        current_masks = []

        assert self.tokenizer.pad_token_id is not None
        pad_token_id = self.tokenizer.pad_token_id

        # iterate over all slices
        for i in range(len(examples["input_ids"])):
            seq_ids = examples["input_ids"][i]
            seq_mask = examples["mask"][i]
            seq_len = len(seq_ids)

            # try to fit it into the current buffer
            if len(current_ids) + seq_len <= self.max_length:
                current_ids.extend(seq_ids)
                current_masks.extend(seq_mask)
            else:
                # doesn't fit; pad the current buffer first
                pad_len = self.max_length - len(current_ids)
                current_ids.extend([pad_token_id] * pad_len)
                current_masks.extend([0] * pad_len)

                # save it
                packed_input_ids.append(current_ids)
                packed_masks.append(current_masks)

                # start a new buffer
                current_ids = list(seq_ids)
                current_masks = list(seq_mask)

        # handle the leftover buffer at the end
        if current_ids:
            pad_len = self.max_length - len(current_ids)
            current_ids.extend([pad_token_id] * pad_len)
            current_masks.extend([0] * pad_len)

            packed_input_ids.append(current_ids)
            packed_masks.append(current_masks)

        return {"input_ids": packed_input_ids, "mask": packed_masks}

    def _compute_token_counts(self, dataset):
        """
        Count tokens for use in Logit Adjustment.
        Only tokens with mask >= 1 are counted (prompt tokens are excluded).
        """
        counter = Counter()
        for batch in dataset.select_columns(["input_ids", "mask"]).iter(batch_size=1000):
            for seq, mask in zip(batch["input_ids"], batch["mask"]):
                counter.update(tok for tok, m in zip(seq, mask) if m >= 1)
        return counter

    def __len__(self):
        return len(self.packed_dataset)

    def __getitem__(self, idx):
        item = self.packed_dataset[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "mask": torch.tensor(item["mask"], dtype=torch.long),
            "dataset_id": torch.tensor(self.dataset_id, dtype=torch.long),
        }

    def get_maskid_to_tokenstr_map(self):
        return self.maskid_to_tokenstr_map

    def get_maskid_to_tokenid_map(self):
        return self.maskid_to_tokenid_map

    def get_token_counts(self):
        return self.token_counts

    def _save_to_cache(self, cache_dir):
        """Saves the packed HF dataset and metadata to disk."""
        os.makedirs(cache_dir, exist_ok=True)
        self.packed_dataset.save_to_disk(os.path.join(cache_dir, "hf_dataset"))

        metadata = {
            "maskid_to_tokenstr_map": self.maskid_to_tokenstr_map,
            "maskid_to_tokenid_map": getattr(self, "maskid_to_tokenid_map", {}),
            "token_counts": self.token_counts,
            "stats": self.stats,
        }
        with open(os.path.join(cache_dir, "metadata.pkl"), "wb") as f:
            pickle.dump(metadata, f)

    def _load_from_cache(self, cache_dir):
        """Loads the packed HF dataset and metadata from disk."""
        self.packed_dataset = load_from_disk(os.path.join(cache_dir, "hf_dataset"))

        with open(os.path.join(cache_dir, "metadata.pkl"), "rb") as f:
            metadata = pickle.load(f)

        self.maskid_to_tokenstr_map = metadata["maskid_to_tokenstr_map"]
        self.maskid_to_tokenid_map = metadata["maskid_to_tokenid_map"]
        self.token_counts = metadata["token_counts"]
        self.stats = metadata["stats"]


class ResampledTapeDataset(Dataset):
    """
    A wrapper around TapeDataset to handle dataset balancing (upsampling/downsampling)
    internally based on a scale factor. This avoids redundant tokenization and caching.
    """

    def __init__(self, dataset, scale, seed=42):
        self.dataset = dataset
        self.scale = scale

        original_count = len(dataset)
        target_count = int(original_count * scale)

        # use a dedicated random instance to ensure reproducibility across environments
        rng = random.Random(seed)
        base_indices = list(range(original_count))

        # generate resampled indices internally based on scale
        if scale == 1.0:
            self.indices = base_indices
        elif scale > 1.0:
            num_full_repeats = int(scale)
            self.indices = base_indices * num_full_repeats
            remainder = target_count - len(self.indices)
            if remainder > 0:
                self.indices.extend(rng.sample(base_indices, k=remainder))
        else:  # scale < 1.0
            self.indices = rng.sample(base_indices, k=target_count)

        # adjust token counts dynamically according to the resampling ratio
        self.token_counts = Counter()
        original_counts = dataset.get_token_counts()
        for k, v in original_counts.items():
            self.token_counts[k] = int(v * scale)

        # adjust the dataset's meta-statistics
        self.stats = dataset.stats.copy()
        self.stats["dataset_samples"] = len(self.indices)
        self.stats["total_conversations"] = int(self.stats["total_conversations"] * scale)
        self.stats["total_tokens"] = int(self.stats["total_tokens"] * scale)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def get_maskid_to_tokenstr_map(self):
        return self.dataset.get_maskid_to_tokenstr_map()

    def get_maskid_to_tokenid_map(self):
        return self.dataset.get_maskid_to_tokenid_map()

    def get_token_counts(self):
        return self.token_counts


class CombinedTapeDataset(ConcatDataset):
    """
    Inherits from ConcatDataset to combine multiple TapeDatasets.
    Adds aggregation of token_counts and unified access to maskid_to_tokenstr_map.
    """

    def __init__(self, datasets):
        super().__init__(datasets)
        self._validate_consistency()
        self.maskid_to_tokenstr_map = datasets[0].get_maskid_to_tokenstr_map()
        self.maskid_to_tokenid_map = datasets[0].get_maskid_to_tokenid_map()

        # Merge stats
        self.stats = {
            "total_conversations": sum(d.stats["total_conversations"] for d in datasets),
            "total_tokens": sum(d.stats["total_tokens"] for d in datasets),
            "dataset_samples": sum(d.stats["dataset_samples"] for d in datasets),
        }
        # Recalculate global averages
        self.stats["average_length"] = self.stats["total_tokens"] / self.stats["total_conversations"]

        # Merge token counts
        self.token_counts = Counter()
        for ds in self.datasets:
            self.token_counts.update(ds.get_token_counts())

    def _validate_consistency(self):
        """
        Ensure the maskid_to_tokenstr_map mapping is consistent across all sub-datasets.
        """
        if not self.datasets:
            return

        first_map = (self.datasets[0].get_maskid_to_tokenstr_map(), self.datasets[0].get_maskid_to_tokenid_map())
        for i, ds in enumerate(self.datasets[1:]):
            current_map = (ds.get_maskid_to_tokenstr_map(), ds.get_maskid_to_tokenid_map())
            if current_map != first_map:
                raise ValueError(
                    f"Inconsistent maskid_to_tokenstr_map or maskid_to_tokenid_map detected between dataset 0 and dataset {i + 1}."
                    f"\nDataset 0: {first_map}"
                    f"\nDataset {i + 1}: {current_map}"
                )

    def get_maskid_to_tokenstr_map(self):
        return self.maskid_to_tokenstr_map

    def get_maskid_to_tokenid_map(self):
        return self.maskid_to_tokenid_map

    def get_token_counts(self):
        return self.token_counts
