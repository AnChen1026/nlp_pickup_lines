"""
train_lora_mac.py

在 Apple Silicon（MPS）上做 LoRA 微調。

train_qlora.py 用 bitsandbytes 的 4-bit 量化，只能在 NVIDIA GPU 上跑。
Mac 沒有 CUDA，所以這支改成：

    Base model 用 float32 載入（不量化）
        ↓
    加上 LoRA adapter
        ↓
    SFTTrainer 只訓練 LoRA 參數
        ↓
    outputs/qwen25_05b_lora/

資料、LoRA 設定、輸出位置都和 train_qlora.py 相同，
所以 infer_lora.py、infer_lora_chat.py、plot_loss.py 可以直接使用。
"""

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
)
from trl import SFTConfig, SFTTrainer


# =============================================================================
# 1. 設定
# =============================================================================

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

TRAIN_FILE = "data/sft/train.jsonl"
VALID_FILE = "data/sft/valid.jsonl"

OUTPUT_DIR = "outputs/qwen25_05b_lora"

MAX_LENGTH = 768
EPOCHS = 5
LEARNING_RATE = 1e-4

# 語料只有兩百多筆，batch 太大的話一個 epoch 只更新一兩次。
# 這裡用 4 × 2 = 8 筆更新一次。
PER_DEVICE_TRAIN_BATCH_SIZE = 4
PER_DEVICE_EVAL_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2

LOGGING_STEPS = 5
CHECK_STEPS = 20
CHECKPOINT_LIMIT = 2
EARLY_STOPPING_PATIENCE = 3

# 和 train_qlora.py 相同。
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = "all-linear"

EOS_TOKEN = "<|im_end|>"
SEED = 42


# =============================================================================
# 2. 主程式
# =============================================================================

def main():
    if not torch.backends.mps.is_available():
        raise RuntimeError("找不到 MPS。這支程式是給 Apple Silicon Mac 用的。")

    print(f"PyTorch：{torch.__version__}")
    print("Device：mps")

    data = load_dataset(
        "json",
        data_files={
            "train": TRAIN_FILE,
            "validation": VALID_FILE,
        },
    )
    print(data)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # MPS 上用 float32 最穩定；0.5B 模型大約 2 GB，記憶體足夠。
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float32,
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    args = SFTConfig(
        output_dir=OUTPUT_DIR,
        max_length=MAX_LENGTH,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=CHECK_STEPS,
        save_strategy="steps",
        save_steps=CHECK_STEPS,
        save_total_limit=CHECKPOINT_LIMIT,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # 只對 assistant 的回答算 loss。
        completion_only_loss=True,
        eos_token=EOS_TOKEN,
        report_to="none",
        seed=SEED,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        processing_class=tokenizer,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=EARLY_STOPPING_PATIENCE
            )
        ],
    )

    trainer.train()

    # load_best_model_at_end=True，這裡存的是 eval_loss 最低的那一個。
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    trainer.state.save_to_json(f"{OUTPUT_DIR}/trainer_state.json")

    print(f"best checkpoint：{trainer.state.best_model_checkpoint}")
    print(f"best eval_loss：{trainer.state.best_metric}")
    print(f"已儲存到 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
