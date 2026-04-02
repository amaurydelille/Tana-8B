from tana_modeling import Tana, MAX_LEN
import torch
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
from torch.utils.data import IterableDataset
from datasets import load_dataset
from typing import Any, Optional
import csv
import os
import shutil
import tempfile
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from safetensors.torch import save_file
from huggingface_hub import HfApi
import deepspeed

TOKENIZER_ID = "mistralai/Mistral-7B-v0.1"
TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_ID)
if TOKENIZER.pad_token is None:
    TOKENIZER.pad_token = TOKENIZER.eos_token
VOCAB_SIZE = TOKENIZER.vocab_size

collator = DataCollatorForLanguageModeling(
    tokenizer=TOKENIZER,
    mlm=False
)


def collate_lm_batch(batch: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = TOKENIZER(
        list(batch),
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"]
    labels = input_ids.clone()
    labels[labels == TOKENIZER.pad_token_id] = -100
    return input_ids, labels

class TanaDataset(IterableDataset):
    def __init__(
        self,
        dataset_id: str,
        dataset_config: Optional[str] = None,
        dataset_split: Optional[str] = None,
        rank: int = 0,
        world_size: int = 1
    ) -> None:
        if dataset_config is None:
            dataset_config = "default"
        if dataset_split is None:
            dataset_split = "train"

        self.dataset_id = dataset_id
        self.dataset_config = dataset_config
        self.dataset_split = dataset_split
        ds = load_dataset(
            dataset_id,
            dataset_config,
            split=dataset_split,
            streaming=True,
        )

        if world_size > 1:
            ds = ds.shard(num_shards=world_size, index=rank)
        self.dataset = ds

    def __iter__(self):
        for sample in self.dataset:
            text = sample.get("text")
            if text is not None:
                yield text

class CSVLogger:
    def __init__(self, csv_file: str) -> None:
        self.csv_file = csv_file

    def csv_logger(self, metrics: dict) -> None:
        with open(self.csv_file, "a") as f:
            writer = csv.writer(f)
            writer.writerow([metrics["batch"], metrics["cross_entropy_loss"], metrics["auxiliary_loss"], metrics["gradient_norm"], metrics["gradient_variance"], metrics["learning_rate"]])

    def compute_metrics(self, batch: int, loss: torch.Tensor, auxiliary_loss: torch.Tensor, gradient_norm: torch.Tensor, gradient_variance: torch.Tensor, learning_rate: float) -> dict:
        return {
            "batch": batch,
            "cross_entropy_loss": loss.item(),
            "auxiliary_loss": auxiliary_loss.item(),
            "gradient_norm": gradient_norm.item(),
            "gradient_variance": gradient_variance.item(),
            "learning_rate": learning_rate,
        }

class Trainer:
    def __init__(
        self, 
        model: Tana, 
        train_dataloader: DataLoader, 
        device: str,
        hf_key: str,
        hf_model_id: str,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        epochs: int = 10,
        csv_logger: Optional[CSVLogger] = None,
        csv_file: str = "metrics.csv",
        args: Optional[Any] = None,
        local_rank: int = 0,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.model = model
        self.train_dataloader = train_dataloader
        self.epochs = epochs
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.device = device
        self.model.to(self.device)
        self.scaler = torch.amp.GradScaler()
        self.csv_logger = csv_logger if csv_logger is not None else CSVLogger(csv_file)
        self.csv_file = csv_file
        self.hf_key = hf_key
        self.hf_model_id = hf_model_id
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size

        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
        self.model_engine, self.optimizer, _, _ = deepspeed.initialize(
            args=args,
            model=model,
            optimizer=optimizer,
            model_parameters=parameters,
        )

    def _save_and_upload_safetensors(self) -> None:
        checkpoint_dir = os.path.join(os.getcwd(), "deepspeed_ckpt_merge")
        os.makedirs(checkpoint_dir, exist_ok=True)
        tag = "hf_upload"
        self.model_engine.save_checkpoint(checkpoint_dir, tag)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        if self.rank != 0:
            return
        from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

        state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir, tag)
        fd, path = tempfile.mkstemp(suffix=".safetensors")
        os.close(fd)
        try:
            save_file(state_dict, path)
            api = HfApi(token=self.hf_key)
            api.create_repo(self.hf_model_id, token=self.hf_key, exist_ok=True)
            api.upload_file(
                path_or_fileobj=path,
                path_in_repo="model.safetensors",
                repo_id=self.hf_model_id,
                token=self.hf_key,
            )
        finally:
            os.unlink(path)
            shutil.rmtree(checkpoint_dir, ignore_errors=True)


    def _train_epoch(self) -> float:
        self.model_engine.train()
        total_loss = 0.0
        batch_count = 0
        for batch_idx, (data, target) in enumerate(self.train_dataloader):
            batch_count += 1
            data = data.to(self.device)
            target = target.to(self.device)
            logits, auxiliary_loss = self.model_engine(data)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = target[:, 1:].contiguous()
            loss = self.criterion(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            ) + auxiliary_loss
            self.model_engine.backward(loss)
            self.model_engine.step()
            total_loss += loss.item()
            if self.rank == 0:
                lr_ref = self.optimizer
                learning_rate = lr_ref.param_groups[0]["lr"]
                metrics = self.csv_logger.compute_metrics(
                    batch_idx, loss, auxiliary_loss,
                    torch.tensor(0.0), torch.tensor(0.0), learning_rate,
                )
                self.csv_logger.csv_logger(metrics)
        return total_loss / batch_count if batch_count > 0 else 0.0

    def train(self) -> None:
        try:
            for epoch in range(self.epochs):
                train_loss = self._train_epoch()
                if self.rank == 0:
                    print(f"Epoch {epoch+1}/{self.epochs}, Loss: {train_loss:.4f}")
            self._save_and_upload_safetensors()
        except (KeyboardInterrupt, Exception):
            self._save_and_upload_safetensors()
            raise