# Question Answering using BERT

Dataset:
- [IndoNLU FacQA](https://github.com/IndoNLP/indonlu/tree/master/dataset/facqa_qa-factoid-itb)


Assuming the terminal workspace under `qa_machine` folder.

Set config in `config.yaml`. 
To conduct LoRA training, in `config.yaml` set `lora_enable=True` and fill lora configuration. to use the model you need to merge the base model tih the adapter weights.
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

- `predict.py` is an example script for prediction 
```bash
python predict.py -f <path/to/input_file.json> -o <path/to/output_file.jsonl>
```

example of input file can be found in `example_io/example_input.json`