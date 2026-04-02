# TANA


## Training
Tana was trained on the following constants:
- Batch size: 8
- $d_{model} = 1024$
- $n_{heads} = 16$
- $d_{hidden} = 1024$
- $n_{experts} = 16$
- $top_k = 8$
- $vocab_{size} = 50000$
- $device = "mps"

You can train Tana on your own dataset using the following command:
```bash
python main.py --epochs 10 --batch_size 8 --hf_key <your_hf_key> --hf_model_id <your_hf_model_id> --hf_dataset_id <your_hf_dataset_id> --hf_dataset_split <your_hf_dataset_split>
```
