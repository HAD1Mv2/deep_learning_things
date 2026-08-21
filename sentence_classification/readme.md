# Fine Tuning Transformers model for Sentence Classificationn Task

This repo contains two methods of fine tuning:

- Full parameter 
- LoRA 

Dataset:

- [Indonlu smsa_doc-sentiment-prosa](https://github.com/IndoNLP/indonlu/tree/master/dataset/smsa_doc-sentiment-prosa)

## Full paramaters fine tuning

Training is done under gradient accumulations setting utilizing `accelerate` package from Huggingface 🤗 .

Assuming the working space under `sentence_classification folder`.

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

Check bash script to modify the input file path.

input file example: `sentence_classification/example_data/input_file_bulk_prediction_example.tsv`

## LoRA fine tuning

With LoRA method, we can fine tune larger models compared to the full parameter tuning. This training process in this scripts also done in gradient accumulation setting.

The scripts can be found under `lora_training` folder.

Assuming the termminal workspace under `lora_training` folder.

- To modify the configuration, set the value in `config_lora.yaml`
- For Training run
    ``` bash
    python train_lora.py
    ```
- Test
    ``` bash
    python test_lora.py
    ```
- Merge base model with the adapter
    ```bash 
    python merge_model_adapter.py
    ```
