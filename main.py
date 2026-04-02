import argparse
import json

from huggingface_hub import HfApi
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import IterableDataset as TorchIterableDataset
from tana_modeling import Tana, TOKENIZER
from trainer import Trainer, TanaDataset, VOCAB_SIZE, collate_lm_batch
import deepspeed
import os

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    print(json.dumps(args.config, indent=4))
    print(json.dumps(getattr(args, "deepspeed_config", None), indent=4))

    deepspeed.init_distributed()
    local_rank = args.local_rank if args.local_rank >= 0 else int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    
    print(f"Local rank: {local_rank}, World size: {world_size}, Rank: {rank}")

    with open(args.config, "r") as f:
        config = json.load(f)

    model_architecture = config["model_architecture"]
    training_parameters = config["training_parameters"]

    if rank == 0:
        hf_identity = HfApi(token=training_parameters["hf_key"]).whoami()
        print(json.dumps(hf_identity, indent=4))

    device_str = f"cuda:{local_rank}"
    model = Tana(
        n_decoders=model_architecture["n_decoders"],
        d_model=model_architecture["d_model"],
        n_heads=model_architecture["n_heads"],
        d_hidden=model_architecture["d_hidden"],
        n_experts=model_architecture["n_experts"],
        top_k=model_architecture["top_k"],
        vocab_size=VOCAB_SIZE,
        device=device_str,
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    train_dataset = TanaDataset(
        dataset_id=training_parameters["hf_dataset_id"],
        dataset_config=training_parameters.get("hf_dataset_config"),
        dataset_split=training_parameters.get("hf_dataset_split"),
        rank=rank,
        world_size=world_size
    )

    num_workers = training_parameters.get("num_workers", 2)
    if isinstance(train_dataset, TorchIterableDataset):
        num_workers = 0

    data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=training_parameters["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=True,
        collate_fn=collate_lm_batch,
    )

    trainer = Trainer(
        model=model,
        train_dataloader=data_loader,
        device=device_str,
        hf_key=training_parameters["hf_key"],
        hf_model_id=training_parameters["hf_model_id"],
        learning_rate=training_parameters["learning_rate"],
        epochs=training_parameters["epochs"],
        args=args,
        local_rank=local_rank,
        rank=rank,
        world_size=world_size
    )

    trainer.train()

    if rank == 0:
        print(model.generate(TOKENIZER, "Salut ca va ?"))

if __name__ == "__main__":
    main()
