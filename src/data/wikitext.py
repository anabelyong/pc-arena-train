import os
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from torchvision.transforms import v2
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk

from src.utils import instantiate_from_config

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROCESSED_DATA_DIR = os.path.join(REPO_ROOT, "processed_wikitext_103")

class WikiTextDataset(Dataset):
    def __init__(self, max_seq_len, tokenizer_name="gpt2", base_path_data=PROCESSED_DATA_DIR, train=True, transform_fns=None):
        self.train = train
        self.max_seq_len = max_seq_len
        self.tokenizer_name = tokenizer_name
        self.base_path_data = base_path_data

        split = "train" if self.train else "validation"
        
        safe_tokenizer_name = "GPT2Tokenizer"
        processed_data_path = os.path.join(self.base_path_data, f"[{self.max_seq_len}]-[{safe_tokenizer_name}]", split)
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        is_main_process = int(os.environ.get("RANK", 0)) == 0
        print(processed_data_path)
        
        if is_main_process and not os.path.exists(processed_data_path):
            print(f"Processing and caching dataset to {processed_data_path}...")
            
            def tokenize_function(examples):
                return tokenizer(examples["text"], truncation=False)

            def group_texts(examples):
                concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
                total_length = len(concatenated_examples[list(examples.keys())[0]])
                total_length = (total_length // self.max_seq_len) * self.max_seq_len
                result = {
                    k: [t[i : i + self.max_seq_len] for i in range(0, total_length, self.max_seq_len)]
                    for k, t in concatenated_examples.items()
                }
                return result

            dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
            dataset = dataset.filter(lambda example: len(example['text'].strip()) > 0)
            
            tokenized_dataset = dataset.map(
                tokenize_function, 
                batched=True, 
                num_proc=max(1, os.cpu_count() // 2),
                remove_columns=["text"]
            )
            
            grouped_dataset = tokenized_dataset.map(
                group_texts, 
                batched=True, 
                num_proc=max(1, os.cpu_count() // 2)
            )
            
            grouped_dataset.save_to_disk(processed_data_path)
            print("Dataset processing and caching complete.")

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        
        print(f"Loading cached dataset from {processed_data_path} for split: {split}")
        self.data = load_from_disk(processed_data_path)
        print(f"Successfully loaded {len(self.data)} samples for split: {split}")

        if transform_fns is not None:
            transforms = []
            for transform_fn in transform_fns:
                transforms.append(instantiate_from_config(transform_fn))
            self.transforms = v2.Compose(transforms)
        else:
            self.transforms = v2.Identity()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = torch.tensor(self.data[index]['input_ids'], dtype=torch.long)
        return self.transforms(sample)


def get_wikitext_dataset(max_seq_len, split = "train", tokenizer = None):
    if tokenizer is None:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // max_seq_len) * max_seq_len
        result = {
            k: [t[i : i + max_seq_len] for i in range(0, total_length, max_seq_len)]
            for k, t in concatenated_examples.items()
        }
        return result

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=False)

    full_path = os.path.join(PROCESSED_DATA_DIR, f"[{max_seq_len}]-[{tokenizer.__class__.__name__}]", split)
    print(full_path)
    # Ensure only the main process handles dataset creation
    if int(os.environ.get("RANK", 0)) == 0 and not os.path.exists(full_path):
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
        dataset = dataset.filter(lambda example: len(example['text'].strip()) > 0)
        tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=64, remove_columns=["text"])
        grouped_dataset = tokenized_dataset.map(group_texts, batched=True, num_proc=64)
        grouped_dataset.save_to_disk(full_path)


if __name__ == "__main__":
    get_wikitext_dataset(128, "train")
    get_wikitext_dataset(128, "validation")