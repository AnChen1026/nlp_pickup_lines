"""
載入 train_qlora.py 儲存的最佳 LoRA adapter，測試模型回答。
若要測試特定 checkpoint，也可以直接把 ADAPTER_DIR 改成 checkpoint 資料夾。

在這個範例裡，我們沒有用到 torchao，若有安裝的話可以移除：
python -m pip uninstall -y torchao
原因：因為我們微調用的是 bitsandbytes NF4，而不是 TorchAO int8，不需要 TorchAO 的 int8 支援。
"""

# 匯入 PyTorch，用 FP16 載入模型並關閉推論梯度。
import torch

# 匯入 PEFT 自動載入器，會依 adapter 設定找到原始 base model。
from peft import AutoPeftModelForCausalLM

# 匯入 tokenizer，用來套用 Qwen chat template。
from transformers import AutoTokenizer


# ===== 推論設定 =====

# 指向 train_qlora.py 的 OUTPUT_DIR；訓練結束後此處會是最佳模型。
ADAPTER_DIR = "outputs/qwen25_05b_lora"  # 也可改成 outputs/.../checkpoint-200 測特定 checkpoint。

# 設定推論時的 system prompt。
# 要和 build_pickup_data.py、infer_lora_chat.py 的 SYSTEM_PROMPT 相同。
SYSTEM_PROMPT = (
    "你是一個情話高手，你會以親切、溫和的口吻來回答使用者的問題。"
    "請用台灣繁體中文，順著上一句接土味情話，不要冒犯對方。"
)

# 設定要測試的使用者問題。
USER_PROMPT = "教我如何透過長相來認識女性朋友。"

# 限制最多產生 256 個新 token，避免教學測試輸出過長。
MAX_NEW_TOKENS = 256  # 一般短對話已足夠。

# 設定 sampling 溫度，0.7 保留一些自然變化但不會過度隨機。
TEMPERATURE = 0.7  # 若要較穩定回答可降低，例如 0.2。


# 定義 LoRA 推論主流程。
def main():
    # 從 adapter 目錄載入 tokenizer。
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)

    # 載入 base model 與 LoRA adapter，並自動放到可用 GPU。
    model = AutoPeftModelForCausalLM.from_pretrained(
        # 指定 LoRA adapter 目錄。
        ADAPTER_DIR,

        # CUDA（例如 Colab T4）用 FP16；
        # Mac 的 MPS 用 FP32，FP16 在 MPS 上生成文字容易不穩定。
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,

        # CUDA 自動放到 GPU。
        # Mac 上 device_map="mps" 會讓程式直接崩潰，
        # 所以先在 CPU 載入，下面再搬到 MPS。
        device_map="auto" if torch.cuda.is_available() else None,
    )

    if not torch.cuda.is_available() and torch.backends.mps.is_available():
        model = model.to("mps")

    # 切換到 evaluation mode，停用 dropout。
    model.eval()

    # 建立標準 system + user 對話輸入。
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": USER_PROMPT}]

    # 套用模型原生 chat template 並轉成 PyTorch tensor。
    inputs = tokenizer.apply_chat_template(
        # 傳入 system 與 user messages。
        messages,

        # 在結尾加入 assistant 開始回答的標記。
        add_generation_prompt=True,

        # 直接將文字 tokenization。
        tokenize=True,

        # 回傳包含 input_ids 等欄位的 dict。
        return_dict=True,

        # 回傳 PyTorch tensor。
        return_tensors="pt",
    ).to(model.device)

    # 推論不需要計算梯度，可降低記憶體使用量。
    with torch.inference_mode():
        # 讓模型產生 assistant 回覆。
        output = model.generate(
            # 傳入 tokenizer 產生的模型輸入。
            **inputs,

            # 限制最多產生的 token 數量。
            max_new_tokens=MAX_NEW_TOKENS,

            # 使用上方指定的 sampling 溫度。
            temperature=TEMPERATURE,

            # 啟用 sampling，讓回答較自然。
            do_sample=True,
        )

    # 只取模型新產生的 token，不包含原始 prompt。
    reply = output[0][inputs["input_ids"].shape[-1]:]

    # 解碼並輸出 assistant 回覆。
    print(tokenizer.decode(reply, skip_special_tokens=True).strip())


# 只有直接執行此檔案時才開始推論。
if __name__ == "__main__":
    # 執行 LoRA 推論主流程。
    main()
