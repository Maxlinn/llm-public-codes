# 模型位置
MODEL_PATH="llama2-7b-hf"
# Alpaca数据位置，可从`https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json`下载
DATA_PATH="./alpaca_data.json"
# 模型存储文件夹
OUTPUT_DIR="./outputs"
# 使用几张显卡来训练
NGPUS=1
# 设置一个不用的端口，用来多卡训练时通信
PORT=23333

torchrun --nproc_per_node=$NGPUS --master_port=$PORT train.py \
    --model_name_or_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --bf16 True \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 3 \
    --optim "adamw_torch" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_target_modules q_proj,v_proj 


# LoRA models are cheap to train, no need to use fsdp
# --fsdp "full_shard auto_wrap" \
# --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \