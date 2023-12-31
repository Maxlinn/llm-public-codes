import torch
import json
from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Trainer, TrainingArguments, HfArgumentParser


@dataclass
class MyArguments:
    # 指向存储LLM模型的位置
    model_name_or_path :str = 'llama2-7b-hf'
    # 使用standford alpaca数据集来训练，可从`https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json`下载
    data_path :str = './alpaca_data.json'


class AlpacaLazyDataset(Dataset):
    '''
    原始的Alpaca数据集形如
    [
        {
            "instruction": "Give three tips for staying healthy.",
            "input": "",
            "output": "1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule."
        },
        {
            "instruction": "What are the three primary colors?",
            "input": "",
            "output": "The three primary colors are red, blue, and yellow."
        }    
    '''
    
    PROMPT_WITH_INPUT = \
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    
    PROMPT_NO_INPUT = \
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
    
    def __init__(self, 
                 tokenizer,
                 alpaca_data_path):
        self.tokenizer = tokenizer
        
        with open(alpaca_data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        example = self.data[index]
        instruction, input, output = example['instruction'], example['input'], example['output']
        
        prompt_template = self.PROMPT_WITH_INPUT if input else self.PROMPT_NO_INPUT
        prompt = prompt_template.format(instruction=instruction, input=input)
        
        prompt_tok = self.tokenizer(f'{self.tokenizer.bos_token}{prompt}', add_special_tokens=False)
        output_tok = self.tokenizer(f'{output}{self.tokenizer.eos_token}', add_special_tokens=False)
        
        prompt_ids, prompt_atts = prompt_tok['input_ids'], prompt_tok['attention_mask']
        output_ids, output_atts = output_tok['input_ids'], output_tok['attention_mask']
        
        input_ids = prompt_ids + output_ids
        attention_mask = prompt_atts + output_atts
        labels = [-100] * len(prompt_ids) + output_ids # -100表示不会被预测的部分
        
        return input_ids, attention_mask, labels


    def collate_fn(self, batch :list):
        # batch是list of examples, 也就是[ [input_ids, attention_mask, labels], [input_ids, attention_mask, labels] ...]
        input_ids_ls = [example[0] for example in batch]
        attention_mask_ls = [example[1] for example in batch]
        labels_ls = [example[2] for example in batch]
        
        # 手动演示pad
        max_length = max(len(input_ids) for input_ids in input_ids_ls)
        for input_ids, attention_mask, labels in zip(input_ids_ls, attention_mask_ls, labels_ls):
            length_to_pad = max_length - len(input_ids)
            
            input_ids.extend([self.tokenizer.pad_token_id] * length_to_pad)
            attention_mask.extend([0] * length_to_pad)
            labels.extend([-100] * length_to_pad)
        
        # Huggingface Trainer要求返回dict, Trainer会利用key为input_ids, attention_mask, labels的项
        # 这些tensor的形状都是[batch_size, seq_len]
        ret = {
            'input_ids': torch.LongTensor(input_ids_ls),
            'attention_mask': torch.LongTensor(attention_mask_ls),
            'labels': torch.LongTensor(labels_ls)
        }
        return ret
    
    
if __name__ == '__main__':
    parser = HfArgumentParser((MyArguments, TrainingArguments))
    my_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(my_args.model_name_or_path, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        # 有的模型可能没有pad_token
        #   pad_token其实不会被预测(通过labels实现)
        #   也不会被模型看到(通过attention_mask实现)
        #   所以设置成什么都可以
        tokenizer.pad_token = tokenizer.unk_token
    
    dataset = AlpacaLazyDataset(tokenizer=tokenizer, alpaca_data_path=my_args.data_path)
    
    torch_dtype = 'auto'
    if training_args.bf16:
        torch_dtype = torch.bfloat16
    print(f'Loading model with precision: {torch_dtype}')
    
    model = AutoModelForCausalLM.from_pretrained(my_args.model_name_or_path, trust_remote_code=True, torch_dtype=torch_dtype)
    
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        eval_dataset=None,
        args=training_args,
        data_collator=dataset.collate_fn
    )
    
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)
