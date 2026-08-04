import itertools
import torch
import contextlib
import math
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq
from accelerate import Accelerator
from utils import SummarizeEncoder, load_config, instantiate_logger, get_lr


logger = instantiate_logger("Training")

logger.info("Load Train Config")
config = load_config("config.yaml", "train")


# fit function for gradient accumulation and DDP training
def fit(num_epochs, num_train_batch_per_epoch, num_dev_batch_per_epoch, gradient_accumulation_step, model, train_dataset, train_loader, dev_loader, opt, loss_function):
    
    min_val_loss = 999
    training_iterator = itertools.cycle(train_loader)
    remainder = num_train_batch_per_epoch % gradient_accumulation_step
    remainder = remainder if remainder !=0 else gradient_accumulation_step
    total_train_updates = math.ceil(num_train_batch_per_epoch/gradient_accumulation_step)

    for epoch in range(num_epochs):

        # insert new seed using epoch
        train_dataset.set_epoch(epoch)
        model.train()
        total_train_loss = 0.0
        total_train_tokens = 0

        train_updates_pbar = tqdm(range(total_train_updates))
        for update_step in train_updates_pbar:
            if update_step == total_train_updates:
                break

            batch_samples = []
            num_batches_in_step = gradient_accumulation_step if update_step != (total_train_updates-1) else remainder
            for _ in range(num_batches_in_step):
                batch_samples += [next(training_iterator)]

            # get local num items in batch
            local_num_items_in_batch = sum([(batch["labels"].ne(-100)).sum() for batch in batch_samples])
            # to compute it correctly in a multi-device DDP training, we need to gather the total number of items in full batch
            num_items_in_batches = accelerator.gather(local_num_items_in_batch).sum().item()

            for i, batch in enumerate(batch_samples):
                if (i < len(batch_samples)-1 and accelerator.num_processes >1):
                    ctx = model.no_sync
                else:
                    ctx = contextlib.nullcontext

                with ctx():
                    input_ids, attention_mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
                    outputs = model(input_ids = input_ids, attention_mask = attention_mask, labels = labels)
                    loss = loss_function(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))

                    loss = (loss * gradient_accumulation_step * accelerator.num_processes) / num_items_in_batches

                    accelerator.backward(loss)

            opt.step()
            opt.zero_grad()

            total_train_loss += (loss*num_items_in_batches)
            total_train_tokens += num_items_in_batches
            global_train_loss_avg = total_train_loss/total_train_tokens if total_train_tokens>0 else 0.0
            train_updates_pbar.set_description("(Epoch {}) TRAIN LOSS:{:.4f} LR:{:.8f}".format((epoch+1), global_train_loss_avg, get_lr(opt)))


        model.eval()
        pbar = tqdm(dev_loader, leave=True, total=num_dev_batch_per_epoch)
        with torch.no_grad():
            total_dev_loss = 0.0
            total_dev_tokens = 0
            for i, batch in enumerate(pbar):
                if i == num_dev_batch_per_epoch:
                    break
                input_ids, attention_mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
                
                outputs = model(input_ids = input_ids, attention_mask = attention_mask, labels = labels)
                dev_loss = loss_function(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))

                local_dev_tokens = (batch["labels"].ne(-100)).sum()

                total_dev_loss += accelerator.gather_for_metrics(dev_loss).sum().item()
                total_dev_tokens += accelerator.gather_for_metrics(local_dev_tokens).sum().item()

                dev_loss_avg = total_dev_loss/total_dev_tokens
                pbar.set_description("(Epoch {}) DEV LOSS:{:.4f}".format((epoch+1), dev_loss_avg))

            global_dev_loss_avg = total_dev_loss/total_dev_tokens if total_dev_tokens > 0 else 0.0
            # we save model with the best val loss  
            if global_dev_loss_avg < min_val_loss:
                min_val_loss = global_dev_loss_avg
                model.save_pretrained(config.save_trained_model_path)    


if __name__ == "__main__":

    logger.info("Load Data")
    data_files = {"train": config.train_data_paths, 
                  "dev": config.dev_data_paths 
                  }

    dataset = load_dataset('json', data_files=data_files, streaming=True)

    logger.info("Begin Data Preparation")
    train_dataset = dataset["train"]
    dev_dataset = dataset["dev"]

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_checkpoint)
    summarizer_encoder = SummarizeEncoder(tokenizer = tokenizer, encoder_max_len=config.encoder_max_len)

    columns_remove = list(train_dataset.features.keys())
    # preprocess/map the dataset using the encode function 

    train_dataset= train_dataset.map(summarizer_encoder.encode, remove_columns=columns_remove)
    train_dataset= train_dataset.shuffle(seed=config.seed, buffer_size=config.buffer_size)
    dev_dataset= dev_dataset.map(summarizer_encoder.encode, remove_columns=columns_remove)

    logger.info("Begin Training Preparation")
    # Wrap data using dataloader since we will train the model by inputing the data batch by batch
    # its not possible to input all data at once when training, since there is memory limitation on GPU 

    # set data collator for seq2seq, datacollator automatically convert encoding into pytorch tensor, don't need to convert encodings in dataset(train_ds, val_ds)
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer)

    train_dl = DataLoader(train_dataset, batch_size=config.batch_size, collate_fn=data_collator)
    dev_dl = DataLoader(dev_dataset, batch_size=config.batch_size, collate_fn=data_collator)

    # Set accelerator
    accelerator = Accelerator(gradient_accumulation_steps=config.gradient_accumulation_step)

    # load model
    model = T5ForConditionalGeneration.from_pretrained(config.model_checkpoint)

    # set optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # load model , opt, dataloader to optimizer, optional: you can also load scheduler
    model, optimizer, train_dl, dev_dl = accelerator.prepare(model, optimizer, train_dl, dev_dl)

    # Since the training setting is gradient accumulation, we set loss reduction as 'sum', and we exclude the pad token (-100) in loss calculation
    loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='sum')

    logger.info("Begin Training!!!")
    fit(num_epochs = config.num_epochs, 
        num_train_batch_per_epoch = config.num_train_batch_per_epoch , 
        num_dev_batch_per_epoch = config.num_dev_batch_per_epoch ,
        gradient_accumulation_step = config.gradient_accumulation_step,
        model = model,
        train_dataset= train_dataset, 
        train_loader = train_dl, 
        dev_loader = dev_dl, 
        opt = optimizer,
        loss_function=loss_fct
        )

    # save tokenizer at the end of training process
    tokenizer.save_pretrained(config.save_trained_model_path)

    logger.info("Done")