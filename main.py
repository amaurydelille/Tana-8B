import argparse
import json

from huggingface_hub import HfApi
from torch.utils.data.dataloader import DataLoader
from tana_modeling import Tana, TOKENIZER
from trainer import Trainer, TanaDataset, VOCAB_SIZE, collate_lm_batch

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    print(json.dumps(args.config, indent=4))

    with open(args.config, "r") as f:
        config = json.load(f)

    model_architecture = config["model_architecture"]
    training_parameters = config["training_parameters"]

    hf_identity = HfApi(token=training_parameters["hf_key"]).whoami()
    print(json.dumps(hf_identity, indent=4))

    model = Tana(
        n_decoders=model_architecture["n_decoders"],
        d_model=model_architecture["d_model"],
        n_heads=model_architecture["n_heads"],
        d_hidden=model_architecture["d_hidden"],
        n_experts=model_architecture["n_experts"],
        top_k=model_architecture["top_k"],
        vocab_size=VOCAB_SIZE,
        device=model_architecture["device"],
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    train_dataset = TanaDataset(
        dataset_id=training_parameters["hf_dataset_id"],
        dataset_config=training_parameters.get("hf_dataset_config"),
        dataset_split=training_parameters.get("hf_dataset_split"),
    )


    num_workers = training_parameters.get("num_workers", 2)
    data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=training_parameters["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=True,
        collate_fn=collate_lm_batch,
    )

    trainer = Trainer(
        model=model,
        train_dataloader=data_loader,
        device=model_architecture["device"],
        hf_key=training_parameters["hf_key"],
        hf_model_id=training_parameters["hf_model_id"],
        learning_rate=training_parameters["learning_rate"],
        epochs=training_parameters["epochs"],
    )

    trainer.train()

    print(model.generate(TOKENIZER, "politique occidentale"))

if __name__ == "__main__":
    main()
