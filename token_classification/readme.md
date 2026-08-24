# Token Classification using BERT

Dataset:
- [IndoNLU NERGrit](https://github.com/IndoNLP/indonlu/tree/master/dataset/nergrit_ner-grit)


Assuming the terminal workspace under `token_classification` folder.

Create workdirs first:
Run:
```bash
python init_workdir.py
```

Set config in `config.yaml`. 
To conduct LoRA training, in `config.yaml` set `lora_enable=True` and fill lora configuration.
If not training with LoRA, the saved model is full params model, you can use it directly for test or prediction.


- For training model, in terminal run:

```bash
python train.py
```

- Merge base model with the LoRA adapter
```bash 
python merge_model_adapter.py
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