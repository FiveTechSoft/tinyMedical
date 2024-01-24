import os
import shutil
import torch
import warnings

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
 
warnings.filterwarnings( "ignore" )

baseModel = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
peft      = "./trained"
outdir    = "./tinyMedical"

print( f"Loading base model: {baseModel}")
model = AutoModelForCausalLM.from_pretrained(
    baseModel,
    torch_dtype = torch.float16,
    device_map = "cuda"
)

print( f"Loading PEFT: {peft}" )
model = PeftModel.from_pretrained( baseModel, peft )
print( "Running merge_and_unload" )
model = model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained( baseModel )

if os.path.isdir( outdir ):
    shutil.rmtree( outdir )

model.save_pretrained( outdir )
tokenizer.save_pretrained( outdir )
print( f"Model saved to {outdir}" )