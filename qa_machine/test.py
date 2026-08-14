import copy
import torch
import evaluate
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from utils import instantiate_logger, load_config, get_default_device


logger = instantiate_logger("Test")

logger.info("Load Test Config")
config = load_config("config.yaml", "test")

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.model_checkpoint)

# encode function for encoding data
def encode(example, encoder_max_len=512):
    
    texts = copy.copy(example['passage'])
    questions= copy.copy(example['question'])
    answers = copy.copy(example['seq_label'])
    answers_text = [None for i in range(len(texts))]
    
    # since the data type in string, we need to convert the data into list
    for i in range(len(texts)):
        t = texts[i].strip("']['").split("', '")
        a = answers[i].strip("']['").split("', '")
        q = questions[i].strip("']['").split(", ")
        q = [b.strip('\'"') for b in q]
        
        if len(t)!=len(a):
            t = texts[i].strip("']['").split(", ")
            t_swap = []
            for b in t:
                if b[0] == '"':
                    t_swap.append(b.strip('"'))
                else:
                    t_swap.append(b.strip("\'"))
            t = t_swap
            
        assert len(t)==len(a)
        
        answers[i] = a
        texts[i] = t
        questions[i] = q
        answers_text[i] = " ".join(list(np.array(t)[np.array(a) != 'O']))
        

    # encode after converting the data
    encoder_inputs = tokenizer(questions, texts, is_split_into_words=True, truncation="only_second", max_length=encoder_max_len, padding=False, 
                               return_overflowing_tokens=True, return_offsets_mapping=True, stride=128)

    # for evaluation purpose we only need model inputs and true answer as string
    encoder_inputs["true answer"]= answers_text
    
    return encoder_inputs


if __name__ == "__main__":

    logger.info("Prepare dataset")
    data_files = {"test": config.test_data}

    dataset = load_dataset('csv', data_files=data_files)

    # preprocess dataset, encode function will be applied on the fly
    dataset.set_transform(encode)

    logger.info("Test preparation")
    n_best_size = config.n_best_size
    max_answer_length = config.max_answer_length
    device = get_default_device()

    model = AutoModelForQuestionAnswering.from_pretrained(config.model_checkpoint).to(device)

    logger.info("Begin test data prediction")
    predicted_answers =[]
    true_answers = []

    pbar = tqdm(dataset['test'], leave=True, total=len(dataset['test']) )
    model.eval()
    for i, example in enumerate(pbar):

        attention_mask = torch.tensor([example['attention_mask']]).to(device)
        input_ids = torch.tensor([example['input_ids']]).to(device)
        token_type_ids = torch.tensor([example['token_type_ids']]).to(device)

        with torch.no_grad():
            output = model(attention_mask = attention_mask, input_ids = input_ids, token_type_ids=token_type_ids)

        start_logits = output.start_logits[0].cpu().numpy()
        end_logits = output.end_logits[0].cpu().numpy()
        offset_mapping = example["offset_mapping"]

        context = example['input_ids']

        # Gather the indices the best start/end logits:
        start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
        end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()

        valid_answers = []
        for start_index in start_indexes:
            for end_index in end_indexes:
                # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                # to part of the input_ids that are not in the context.
                if (
                    start_index >= len(offset_mapping)
                    or end_index >= len(offset_mapping)
                    or offset_mapping[start_index] is None
                    or offset_mapping[end_index] is None
                ):
                    continue
                # Don't consider answers with a length that is either < 0 or > max_answer_length.
                if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                    continue
                #if start_index <= end_index: # We need to refine that test to check the answer is inside the context
                #start_char = offset_mapping[start_index][0]
                #end_char = offset_mapping[end_index][1]
                valid_answers.append(
                    {
                        "score": start_logits[start_index] + end_logits[end_index],
                        "text": tokenizer.decode(context[start_index: end_index], skip_special_tokens=True),
                        "start_idx": start_index,
                        "end_idx": end_index
                    }
                )

        valid_answers = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[:n_best_size]
        
        try:
            predicted_answers.append({'id': f'{i}', 'prediction_text': valid_answers[0]['text']})
        except:
            predicted_answers.append({'id': f'{i}', 'prediction_text': ""})
            print(valid_answers)
            print(start_indexes)
            print(end_indexes)

        true_answers.append({'id': f'{i}', 'answers': {'text': [example['true answer']], 'answer_start': [0]}})

    logger.info("Calculate evaluation metric")
    metric = evaluate.load("squad")
    eval_score = metric.compute(predictions=predicted_answers, references=true_answers)

    print(eval_score)
    logger.info("Script end")
 