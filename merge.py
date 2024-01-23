import os
import shutil
import torch
import argparse
import warnings

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from os import system
 
def main():
 
    system( "clear" )
    warnings.filterwarnings( "ignore" )

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--lora", type=str)
    parser.add_argument("--out_dir", type=str, default="./model") # leave this
    args = parser.parse_args()

    args.model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    args.lora = "./TinyLlama"
    args.out_dir = "Model"
 
    print(f"Loading base model: {args.model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
#        load_in_8bit = True,
        torch_dtype = torch.float16,
        device_map = "cuda"
    )
 
    print( f"Loading PEFT: {args.lora}" )
    model = PeftModel.from_pretrained( base_model, args.lora )
    print( f"Running merge_and_unload" )
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained( args.model )
 
    if os.path.isdir( args.out_dir ):
        shutil.rmtree( args.out_dir )

    model.save_pretrained( f"{args.out_dir}" )
    tokenizer.save_pretrained( f"{args.out_dir}" )
    print( f"Model saved to {args.out_dir}" )
 
if __name__ == "__main__" :
    main()