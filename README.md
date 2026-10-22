# 作業：土味情話多輪對話專家

用 QLoRA 微調小型語言模型，建立一個可以多輪對話的「土味情話」專家。

## 這個模型在做什麼

使用者丟出一個情境，模型用親切、溫和的台灣繁體中文接下一句土味情話。對話至少可以連續 2 輪。

system prompt：

```text
你是一個情話高手，你會以親切、溫和的口吻來回答使用者的問題。請用台灣繁體中文，順著上一句接土味情話，不要冒犯對方。
```

## 基礎模型

- [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)

## 執行方法

在 `cases/llm_finetune/` 依序執行。訓練需要 NVIDIA GPU，請用 Colab，不要在沒有 CUDA 的 Mac 上跑 `train_qlora.py`。

```bash
python build_pickup_data.py
python prepare_chat_data.py
python train_qlora.py
python infer_lora_chat.py
```

`build_pickup_data.py` 會寫出 `data/chat.jsonl`，`prepare_chat_data.py` 再把它切成 `data/sft/train.jsonl`、`data/sft/valid.jsonl`。

多輪測試時可以先輸入：

```text
教我如何透過長相來認識女性朋友。
長得像你朋友？
```

輸入 `exit` 結束。

## 語料範例

下面是訓練語料裡的一段，不是模型已經生成的結果。

```text
你：教我如何透過長相來認識女性朋友。
專家：你就跟她說「你長得很像我一個朋友」。
你：長得像你朋友？
專家：沒錯，跟她說「你長得像我下一任女朋友」。
```
