import torch
import numpy as np
from datasets import Dataset
from torch.utils.data import DataLoader
from utils import read_wnut, instantiate_logger, get_default_device, load_config
from transformers import AutoModelForTokenClassification, BertTokenizerFast
from tqdm.auto import tqdm
from seqeval.scheme import IOB2
from seqeval.metrics import classification_report as seqeval_classification_report

logger = instantiate_logger("Token Class Test")

logger.info("load config")
config = load_config("config.yaml", "test")


if __name__ == "__main__":

    # prepare tokenizer and model
    logger.info("Load Tokenizer and Model")
    device = get_default_device()
    tokenizer = BertTokenizerFast.from_pretrained(config.model_checkpoint, do_lower_case=True, max_length = 512)
    model = AutoModelForTokenClassification.from_pretrained(config.model_checkpoint)
    model = model.to(device)

    logger.info("Load test data and preprocess")
    test_texts, test_tags = read_wnut(config.test_data_path)
    test_encoding = tokenizer(test_texts, is_split_into_words=True, return_offsets_mapping=True, max_length=config.encoder_max_len, padding='max_length', truncation=True)
    # offset mapping used in the process of token decoding
    offset_mapping = test_encoding.pop('offset_mapping')
    test_dataset = Dataset.from_dict({'input_ids': test_encoding['input_ids'], 'attention_mask': test_encoding['attention_mask']})
    # we need to format the data into pytorch Tensor, since we use pytorch model for prediction
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'], output_all_columns=True)
    # put the dataset into data loader
    test_dl = DataLoader(test_dataset, config.batch_size, shuffle=False)

    logger.info("Begin predictions")
    pbar = tqdm(test_dl, leave=True, total=len(test_dl))
    with torch.no_grad():
        list_pred_label = []
        for idx, data in enumerate(pbar):
            input_ids, attention_mask = data["input_ids"], data["attention_mask"]
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            output = model(input_ids = input_ids, attention_mask = attention_mask)
            logits = output.logits
            logits_labels = torch.argmax(logits, dim=-1).tolist()

            for label in logits_labels:
                tag = np.array([model.config.id2label[i] for i in label])
                list_pred_label.append(tag)

    final_list_pred_label = []
    for pred_label, map in zip(list_pred_label, offset_mapping):
        arr_map = np.array(map)
        mask = (arr_map[:,0] == 0) & (arr_map[:,1] != 0)
        final_list_pred_label.append(pred_label[mask].tolist())

    logger.info("Evaluate and print report")
    print(seqeval_classification_report(test_tags, final_list_pred_label, mode='strict', scheme=IOB2))



