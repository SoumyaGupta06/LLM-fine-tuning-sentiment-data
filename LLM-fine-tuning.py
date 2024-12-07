import pandas as pd
import datasets
from pprint import pprint
from transformers import AutoTokenizer
from pprint import pprint
import datasets
import tempfile
import logging
import random
import config
import os
import yaml
import time
import torch
import transformers
import pandas as pd
import jsonlines
from utilities import *
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import TrainingArguments
from transformers import AutoModelForCausalLM
from llama import BasicModelRunner
import pandas as pd
import datasets
from pprint import pprint
from transformers import AutoTokenizer
from pprint import pprint
from datasets import DatasetDict
from collections import defaultdict
from datasets import load_dataset
from tqdm import tqdm


# data preparation

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")

hf_dataset = "mteb/tweet_sentiment_extraction"
dataset_sentiment = datasets.load_dataset(hf_dataset)
print(dataset_sentiment["train"][1])

data_train = dataset_sentiment["train"]

prompt_template = """### Input:
{inp}

### Output:"""

num_examples = len(data_train["text"])

finetuning_dataset = []
for i in range(100):
    inp = data_train["text"][i]
    opt = data_train["label_text"][i]
    text_with_prompt_template = prompt_template.format(inp=inp)
    finetuning_dataset.append({"input": text_with_prompt_template, "output": opt})
   

print("One datapoint in the finetuning dataset:")
pprint(finetuning_dataset[0])

#tokenizing data
def tokenize_function(finetuning_dataset):
    if "question" in finetuning_dataset and "answer" in finetuning_dataset:
      text = finetuning_dataset["question"] + finetuning_dataset["answer"]
    elif "input" in finetuning_dataset and "output" in finetuning_dataset:
      text = finetuning_dataset["input"] + finetuning_dataset["output"]
    else:
      text = finetuning_dataset["text"]

    tokenizer.pad_token = tokenizer.eos_token
    tokenized_inputs = tokenizer(
        text,
        return_tensors="np",
        padding=True,
    )

    max_length = min(
        tokenized_inputs["input_ids"].shape[1],
        2048
    )
    tokenized_inputs = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        max_length=max_length
    )
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
    return tokenized_inputs

tokenized_dataset = dataset_sentiment.map(
    tokenize_function,
    batched=True,
    batch_size=1,
    drop_last_batch=True
)
print(tokenized_dataset)
tokenized_dataset.save_to_disk('D:/Projects/tokenized_dataset')


# training

logger = logging.getLogger(__name__)
global_config = None

dataset_path = 'D:/Projects/tokenized_dataset'
use_hf = False

data_train = dataset_sentiment["train"]
data_test = dataset_sentiment["test"]

model_name = "EleutherAI/pythia-70m"

training_config = {
    "model": {
        "pretrained_name": model_name,
        "max_length": 2048
    },
    "datasets": {
        "use_hf": use_hf,
        "path": dataset_path
    },
    "verbose": True
}

base_model = AutoModelForCausalLM.from_pretrained(model_name)

device_count = torch.cuda.device_count()
if device_count > 0:
    logger.debug("Select GPU device")
    device = torch.device("cuda")
else:
    logger.debug("Select CPU device")
    device = torch.device("cpu")

base_model.to(device)

def inference(test_text, model, tokenizer, max_input_tokens=1000, max_output_tokens=100):
    input_ids = tokenizer.encode(
        test_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens
    )

    device = model.device
    generated_tokens_for_test = model.generate(
        input_ids=input_ids.to(device),
        max_length=max_output_tokens
    )

    generated_text_for_test = tokenizer.batch_decode(generated_tokens_for_test, skip_special_tokens=True)
    generated_text_answer = generated_text_for_test[0][
                            len(test_text):]  # removing input prompt from the answer generated

    return generated_text_answer


test_text = data_test[0]['text']
print("Question input (test):", test_text)
print(f"Correct answer from data: {data_test[0]['label_text']}")
print("Model's answer: ")
print(inference(test_text, base_model, tokenizer))

max_steps = len(dataset_sentiment["train"])

trained_model_path = r'D:/Projects/trained_model'
output_dir = trained_model_path

training_args = TrainingArguments(

    learning_rate=1.0e-5,
    num_train_epochs=1,
    max_steps=max_steps,
    per_device_train_batch_size=1,
    output_dir=output_dir,

    overwrite_output_dir=False,
    disable_tqdm=False,
    eval_steps=120,  # Number of update steps between two evaluations
    save_steps=120,
    warmup_steps=1,
    per_device_eval_batch_size=1,  # Batch size for evaluation
    evaluation_strategy="steps",
    logging_strategy="steps",
    logging_steps=1,
    optim="adafactor",
    gradient_accumulation_steps=4,
    gradient_checkpointing=False,

    load_best_model_at_end=True,
    save_total_limit=1,
    metric_for_best_model="eval_loss",
    greater_is_better=False
)

model_flops = (
        base_model.floating_point_ops(
            {
                "input_ids": torch.zeros(
                    (1, training_config["model"]["max_length"])
                )
            }
        )
        * training_args.gradient_accumulation_steps
)

print(base_model)
print("Memory footprint", base_model.get_memory_footprint() / 1e9, "GB")
print("Flops", model_flops / 1e9, "GFLOPs")

data_train = data_train.map(tokenize_function, batched=True, batch_size=1)
data_test = data_test.map(tokenize_function, batched=True, batch_size=1)

trainer = Trainer(
    model=base_model,
    model_flops=model_flops,
    total_steps=max_steps,
    args=training_args,
    train_dataset=data_train,
    eval_dataset=data_test,
)
training_output = trainer.train()

finetuned_model = trainer.model

test_question = data_test[0]['text']
print("Input (test):", test_question)

print("Finetuned model's answer: ")
print(inference(test_question, finetuned_model, tokenizer))

test_answer = data_test[0]['label_text']
print("Target output (test):", test_answer)


# evaluation

dataset = datasets.load_dataset("mteb/tweet_sentiment_extraction")
test_dataset = dataset["test"]
print(test_dataset[0]["text"])
print(test_dataset[0]["label_text"])

model = finetuned_model


# evaluation func
def is_exact_match(a, b):
    return a.strip() == b.strip()

model.eval()

def inference(text, model, tokenizer, max_input_tokens=1000, max_output_tokens=100):
    # Tokenize
    tokenizer.pad_token = tokenizer.eos_token
    input_ids = tokenizer.encode(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens
    )

    # Generate
    device = model.device
    generated_tokens_with_prompt = model.generate(
        input_ids=input_ids.to(device),
        max_length=max_output_tokens
    )

    # Decode
    generated_text_with_prompt = tokenizer.batch_decode(generated_tokens_with_prompt, skip_special_tokens=True)

    # Strip the prompt
    generated_text_answer = generated_text_with_prompt[0][len(text):]

    return generated_text_answer


test_question = test_dataset[0]["text"]
generated_answer = inference(test_question, model, tokenizer)
print(test_question)
print(generated_answer)

answer = test_dataset[0]["label_text"]
print(answer)

exact_match = is_exact_match(generated_answer, answer)
print(exact_match)

n = 10
metrics = {'exact_matches': []}
predictions = []
for i, item in tqdm(enumerate(test_dataset)):
    print("i Evaluating: " + str(item))
    question = item['text']
    answer = item['label_text']

    try:
        predicted_answer = inference(question, model, tokenizer)
    except:
        continue
    predictions.append([predicted_answer, answer])

    exact_match = is_exact_match(predicted_answer, answer)
    metrics['exact_matches'].append(exact_match)

    if i > n and n != -1:
        break
print('Number of exact matches: ', sum(metrics['exact_matches']))

df = pd.DataFrame(predictions, columns=["predicted_answer", "target_answer"])
print(df)

