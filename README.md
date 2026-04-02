# TANA-LLM
![alt text](asset.png)

## Training
Tana was trained on the following constants:
- Batch size: 8
- $d_{model} = 1024$
- $n_{heads} = 16$
- $d_{hidden} = 1024$
- $n_{experts} = 16$
- $top_k = 8$
- $vocab_{size} = 50000$
- $device$ = "mps"

You can train Tana on your own dataset using the following command:
```bash
python main.py --config <your_config_file.json>
```

Tana also supports DeepSpeed for distributed training. To use DeepSpeed, you can run the following command:
```bash
deepspeed main.py --config training_config.json --deepspeed_config ds_config.json
```