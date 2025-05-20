from __future__ import annotations
import argparse
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)
from peft import PeftModel
from torch.utils.data import DataLoader

HF_MODEL_ID = "huggyllama/llama-7b"
ADAPTER_PATH = "/home/jiaxuan/Documents/Projects/slora-plus/S-LoRA/test/test_e2e/finetuning_adapter"
DATA_PATH = Path("/home/jiaxuan/Documents/Projects/slora-plus/S-LoRA/test/test_e2e/finetune_test.csv")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--output_dir", type=Path, default=Path("./lora_finetuned"))
    p.add_argument("--load_8bit", action="store_true")
    return p.parse_args()


def build_dataset(tokenizer, file_path: Path, max_length: int = 128):
    ds = load_dataset("text", data_files=str(file_path), split="train")

    def _tokenise(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_length)

    ds = ds.map(_tokenise, batched=True, remove_columns=["text"], num_proc=1)
    ds.set_format(type="torch")
    return ds


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_ID,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        load_in_8bit=args.load_8bit,
    )

    for p in model.parameters():
        p.requires_grad_(False)

    model = PeftModel.from_pretrained(model, ADAPTER_PATH, is_trainable=True)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    dataset = build_dataset(tokenizer, DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=len(dataset), collate_fn=DataCollatorForLanguageModeling(tokenizer, mlm=False))

    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for epoch in range(args.epochs):
        for batch in dataloader:
            for k in batch:
                batch[k] = batch[k].to(device)

            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch + 1}: Loss = {loss.item():.6f}")

    adapter_save_path = args.output_dir / "finetuned_adapter"
    model.save_pretrained(adapter_save_path, safe_serialization=True)
    print(f"LoRA adapter saved to {adapter_save_path}")


if __name__ == "__main__":
    main()