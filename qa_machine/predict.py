import torch
import argparse
import numpy as np
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from utils import instantiate_logger, load_config, get_default_device, read_json, save_to_jsonl


logger = instantiate_logger("Predict")

logger.info("Load Predict Config")
config = load_config("config.yaml", "predict")

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.model_checkpoint)
device = get_default_device()


def args_parser():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", type=str, default=None, help="a .json file containing input data ex. {'passage': ..., 'question': ...}")
    parser.add_argument("-o", "--outfile", type=str, default=None, help="output file containing list of answers from the model, a .jsonl file.")

    args = parser.parse_args()

    return args


# function for prediction
class QAMachine:
    def __init__(self, model_checkpoint, device, encoder_max_len = config.encoder_max_len, max_answer_length = config.max_answer_length, n_best_size = config.n_best_size):
        self.device = device
        self.encoder_max_len = encoder_max_len
        self.max_answer_length = max_answer_length
        self.n_best_size = n_best_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint).to(self.device)

    # convert raw input into feature vector
    def prepare_features(self, example):

        text = example['passage']
        question = example['question']
            
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.

        tokenized_examples = self.tokenizer(question, text, is_split_into_words=False, truncation="only_second", 
                                            max_length=config.encoder_max_len, padding=False, 
                                            return_overflowing_tokens=True, return_offsets_mapping=True, stride=config.doc_stride)

        return tokenized_examples        
    
    def __call__(self, example):
        
        features  = self.prepare_features(example)

        best_answers =[]
            
        attention_mask = torch.tensor(features['attention_mask']).to(self.device)
        input_ids = torch.tensor(features['input_ids']).to(self.device)
        token_type_ids = torch.tensor(features['token_type_ids']).to(self.device)

        with torch.no_grad():
            output = self.model(attention_mask = attention_mask, input_ids = input_ids, token_type_ids=token_type_ids)


        start_logits = output.start_logits[0].cpu().numpy()
        end_logits = output.end_logits[0].cpu().numpy()
        offset_mapping = features["offset_mapping"][0]

        context = features['input_ids'][0]

        # Gather the indices the best start/end logits:
        start_indexes = np.argsort(start_logits)[-1 : -self.n_best_size - 1 : -1].tolist()
        end_indexes = np.argsort(end_logits)[-1 : -self.n_best_size - 1 : -1].tolist()

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
                if end_index < start_index or end_index - start_index + 1 > self.max_answer_length:
                    continue
                #if start_index <= end_index: # We need to refine that test to check the answer is inside the context
                #start_char = offset_mapping[start_index][0]
                #end_char = offset_mapping[end_index][1]
                valid_answers.append(
                    {
                        "score": start_logits[start_index] + end_logits[end_index],
                        "text": tokenizer.decode(context[start_index: end_index]),
                        "start_idx": start_index,
                        "end_idx": end_index
                    }
                )

        valid_answers = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[:self.n_best_size]
        
        try:
            best_answers.append(valid_answers[0])
        except:
            print(valid_answers)
            print(start_indexes)
            print(end_indexes)

        #return best_answers
        return valid_answers


if __name__ == "__main__":

    input_args = args_parser()

    logger.info("Preparing input")
    qa_input = read_json(input_args.filename)

    logger.info("Create QA agent") 
    device = get_default_device()
    qa_agent = QAMachine(model_checkpoint = config.model_checkpoint, device = device)

    logger.info("QA answering...")
    answer = qa_agent(qa_input)

    logger.info("Saving answers")
    save_to_jsonl(answer, input_args.outfile)
    logger.info("End process")
    