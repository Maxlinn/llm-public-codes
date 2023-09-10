import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-model', type=str, help='path of base model dir.')
    parser.add_argument('--lora-model', type=str, help='path of lora model dir.')
    parser.add_argument('--output-dir', type=str, help='path of dir to save the model')
    parser.add_argument('--no-tokenizer', action='store_true', help='by default, also save tokenizer from base model into output, set this flag to avoid.')
    args = parser.parse_args()
    
    if not args.no_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, use_fast=False)
    
    model = AutoModelForCausalLM.from_pretrained(args.base_model, trust_remote_code=True, torch_dtype='auto')
    peft_model = PeftModel.from_pretrained(model, args.lora_model)
    
    merged = peft_model.merge_and_unload()
    print('merged model, saving...')
    os.makedirs(args.output_dir, exist_ok=True)
    merged.save_pretrained(args.output_dir)
    
    if not args.no_tokenizer:
        tokenizer.save_pretrained(args.output_dir)