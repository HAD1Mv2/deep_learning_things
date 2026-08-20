import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
from utils import instantiate_logger, load_config


# Set logger 
logger = instantiate_logger("Merge model + adapter")

if __name__ == "__main__":

    logger.info("Load Config")
    config = load_config("config_lora.yaml", "merge_model_lora")

    logger.info("Load tokenizer + model + adapter")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_checkpoint)

    base_model = AutoModelForSequenceClassification.from_pretrained(config.base_model_checkpoint)

    model = PeftModel.from_pretrained(base_model, config.adapter_path)

    logger.info("Merge base model with adapter")
    merged_model = model.merge_and_unload()

    logger.info("Save merger model and tokenizer")
    merged_model.save_pretrained(config.merge_model_save_path)
    tokenizer.save_pretrained(config.merge_model_save_path)

    logger.info("End")

