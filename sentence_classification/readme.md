# Sentence Classification Script

Create workdirs first:
Run:
```bash
python init_workdir.py
```

Set config in `config.yaml`

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
python predict.py
```

- For bulk prediction
```bash
bash bulk_predict_script.sh
```
input file example: `sentence_classification/example_data/input_file_bulk_prediction_example.tsv`