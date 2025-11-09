import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

def main():
    model_dir = "/home/haibin/apac25/dpsk8B"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    print(f"Loading local model from {model_dir} on {device} ({dtype})...")

    # ==== 加载 tokenizer ====
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True
    )

    # ==== 加载模型 ====
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    # ==== Prompt ====
    prompt = "解释一下大语言模型中 attention 的工作原理。"

    # ==== 编码 ====
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # ==== 生成 ====
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    # ==== 解码 ====
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    print("\n=== Model Output ===\n")
    print(text)

if __name__ == "__main__":
    main()
