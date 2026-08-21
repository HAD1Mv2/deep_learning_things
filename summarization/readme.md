# Fine-tuning T5 Based model for summarization task

This repo contains two methods of fine tuning:

- Full parameter 
- LoRA 

Dataset:

- [IndoSum](https://arxiv.org/abs/1810.05334)


## Full parameter fine tuning

Run `python init_wortkdir.py` in the terminal to initialize structured folders to run the script, put the data in folder `data`.

We can modify the configuration in `config.yaml` file. Training is under gradient accumulations setting.

Assuming the terminal workdir space is under `summarization` folder.

- For training, run
    ``` bashh
    python train.py
    ```

- Test
    ```bash
    python test.py
    ```

`summarize.py` is a simple example script to use the trained model. To run it

```bash
python summarize.py -f <path/to/text_to_summmarize.txt> -o <path/to/save/summary.txt>
```
Check `input_example/article.txt` for input file example. 

## LoRA fine tuning

With LoRA method, we can fine tune larger models compared to the full parameter tuning. 

The scripts are located under folder `lora_training`. 
Modify the configuration in `config_lora.yaml` .

Assuming the terminal workdir space is under `lora_training` folder. We can run the scripts with the following commands.

- Train
    ```bash
    python train_lora.py
    ```
- Test
    ```bash
    python test_lora.py
    ```
- Merge base model with the lora adapter 
    ```bash
    python merged_model_adapter.py
    ```