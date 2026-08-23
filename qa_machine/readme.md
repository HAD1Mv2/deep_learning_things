# Question Answering using BERT

Dataset:
- [IndoNLU FacQA](https://github.com/IndoNLP/indonlu/tree/master/dataset/facqa_qa-factoid-itb)


Assuming the terminal workspace under `qa_machine` folder.

Set config in `config.yaml`. 
To conduct LoRA training, in `config.yaml` set `lora_enable=True` and fill lora donfiguration. to use the model you need to merge the base model tih the adapter weights.
If not training with LoRA, the saved model is full params model, you can use it directly for test or prediction.

- For training model, in terminal run:

```bash
python train.py
```

- Merge base model with the adapter
```bash 
python merge_model_adapter.py
```

- For test model
```bash
python test.py
```

- Example script for prediction
```bash
python predict.py --text "Your Sentence here"
```