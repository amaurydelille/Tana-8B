from tana_modeling import Tana, MAX_LEN
import torch
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
from torch.utils.data import Dataset
from datasets import load_dataset
from typing import Optional, Literal
import csv
import os
import tempfile
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from safetensors.torch import save_file
from huggingface_hub import HfApi

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

class TanaDataset(Dataset):
    def __init__(
        self,
        dataset_id: str,
        dataset_config: Optional[str] = None,
        dataset_split: Optional[str] = None,
    ) -> None:
        if dataset_config is None:
            dataset_config = "default"
        if dataset_split is None:
            dataset_split = "train"

        self.dataset_id = dataset_id
        self.dataset_config = dataset_config
        self.dataset_split = dataset_split
        self.dataset = load_dataset(
            dataset_id,
            dataset_config,
            split=dataset_split,
            streaming=True,
        )

    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> str:
        return self.dataset[idx]["text"]

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
        device: Literal["cpu", "cuda", "mps"],
        hf_key: str,
        hf_model_id: str,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        epochs: int = 10,
        csv_logger: Optional[CSVLogger] = None,
        csv_file: str = "metrics.csv",
    ) -> None:
        self.model = model
        self.train_dataloader = train_dataloader
        self.epochs = epochs
        muon_params = [p for p in model.parameters() if p.ndim == 2]
        other_params = [p for p in model.parameters() if p.ndim != 2]
        self.optimizer_muon = (
            optim.Muon(muon_params, lr=learning_rate, weight_decay=weight_decay)
            if muon_params
            else None
        )
        self.optimizer_adam = (
            optim.AdamW(other_params, lr=learning_rate, weight_decay=weight_decay)
            if other_params
            else None
        )
        self._optimizers = [
            o for o in (self.optimizer_muon, self.optimizer_adam) if o is not None
        ]
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.device = device
        self.model.to(self.device)
        self.scaler = torch.amp.GradScaler()
        self.csv_logger = csv_logger if csv_logger is not None else CSVLogger(csv_file)
        self.csv_file = csv_file
        self.hf_key = hf_key
        self.hf_model_id = hf_model_id

    def _save_and_upload_safetensors(self) -> None:
        state_dict = {k: v.contiguous().cpu() for k, v in self.model.state_dict().items()}
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

    def _train_epoch(self) -> None:
        self.model.train()
        total_loss = 0.0

        for batch_idx, (data, target) in enumerate(self.train_dataloader):
            
            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=True):
                data, target = data.to(self.device), target.to(self.device)
                logits, auxiliary_loss = self.model(data)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = target[:, 1:].contiguous()
                loss = self.criterion(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                ) + auxiliary_loss

            self.scaler.scale(loss).backward()
            for opt in self._optimizers:
                self.scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            for opt in self._optimizers:
                self.scaler.step(opt)
            self.scaler.update()
            for opt in self._optimizers:
                opt.zero_grad(set_to_none=True)

            total_loss += loss.item()

            params_flat = torch.cat([p.detach().flatten() for p in self.model.parameters()])
            gradient_norm = torch.norm(params_flat)
            gradient_variance = torch.var(params_flat)
            lr_ref = self.optimizer_muon or self.optimizer_adam
            assert lr_ref is not None
            learning_rate = lr_ref.param_groups[0]["lr"]
            metrics = self.csv_logger.compute_metrics(batch_idx, loss, auxiliary_loss, gradient_norm, gradient_variance, learning_rate)
            self.csv_logger.csv_logger(metrics)

        return total_loss / len(self.train_dataloader)

    def train(self) -> None:
        try:
            for epoch in range(self.epochs):
                train_loss = self._train_epoch()
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {train_loss:.4f}")

            self._save_and_upload_safetensors()
        except (KeyboardInterrupt, Exception):
            self._save_and_upload_safetensors()
            raise
