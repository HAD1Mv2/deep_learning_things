# Token Classification using BERT

Create workdirs first:
Run:
```bash
python init_workdir.py
```

Set config in `config.yaml`. 
To conduct LoRA training, in `config.yaml` set `lora_enable=True` and fill lora donfiguration.

- For training model, in terminal run:

```bash
python train.py
```

- For test model

```bash
python test.py
```

- For single sentence prediction
```bash
python predict.py --text "Your Sentence here"
```

- For bulk prediction
```bash
python predict_bulk -f path/to/input_file.txt -o path/for/result.jsonl
```
input file example for bulk predict: `token_classification/example_data/demo.txt`