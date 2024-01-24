import os
import pandas as pd
import json
import torch
import shutil
from os import system
from datetime import datetime
from datasets import load_dataset
from peft import LoraConfig
from peft.utils.other import prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments
)
from trl import SFTTrainer

print( "Loading the dataset..." )
df = pd.read_csv( "medquad.csv" )
df = df.iloc[ :, :2 ]
df.columns = [ "text", 'label' ]

#result = list(df.head(1000).to_json(orient="records"))
result = list( df.to_json( orient="records" ) )
result[0] = '{"json":['
result[-1] = ']'
result.append('}')
result = ''.join(result)
result = result.strip('"\'')
result = json.loads( result )

with open( 'data.json', 'w' ) as json_file:
    json.dump( result, json_file )

modelName = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model = AutoModelForCausalLM.from_pretrained(
    modelName,
    device_map = 'auto',
    trust_remote_code = True,
    token = False,
)
model = prepare_model_for_kbit_training( model )

peft_config = LoraConfig(
    r = 32,
    lora_alpha = 16,
    bias = "none",
    lora_dropout = 0.05, # Conventional
    task_type = "CAUSAL_LM",
)
model.add_adapter( peft_config )
model.config.pretraining_tp = 1
tokenizer = AutoTokenizer.from_pretrained( modelName, trust_remote_code = True, use_fast=False )
tokenizer.pad_token = tokenizer.eos_token

if os.path.isdir( "./temp" ):
    shutil.rmtree( "./temp" )

training_arguments = TrainingArguments(
    output_dir = "./temp",
    num_train_epochs = 4,
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 1,
    optim = "paged_adamw_32bit",
    save_strategy = "epoch",
    logging_steps = 100,
    logging_strategy = "steps",
    learning_rate= 2e-4,
    fp16= False,
    bf16= False,
    group_by_length = True,
    disable_tqdm = False,
    report_to = None
)

model.config.use_cache = False

def formatting_func( example ):
    text = f"### Question: { example['text'] }\n ### Answer: { example['label'] }"
    return text

def generate_and_tokenize_prompt( prompt ):
    return tokenizer( formatting_func( prompt ), truncation = True, max_length = 2048 )

dataset = load_dataset("json", data_files="data.json", field='json', split="train")
dataset = dataset.map( generate_and_tokenize_prompt )

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=2048,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,
)

trainer.train()

if os.path.isdir( "./temp" ):
    shutil.rmtree( "./temp" )

directory = "trained"

if os.path.isdir( directory ):
    shutil.rmtree( directory )

model.save_pretrained( directory )
tokenizer.save_pretrained( directory )
print( f"Model saved '{directory}'." )