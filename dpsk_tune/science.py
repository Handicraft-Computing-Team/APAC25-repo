# profile_dpsk_mem.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.profiler import profile, ProfilerActivity, record_function

MODEL_DIR = "/home/haibin/apac25/dpsk8B"
OUTDIR = "./prof_out"
os.makedirs(OUTDIR, exist_ok=True)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # 推荐的 TF32 配置（A100 上有益，忽略告警即可）
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    print(f"[Info] Loading local model from: {MODEL_DIR}")
    print(f"[Info] Device={device}, dtype={dtype}, outdir={OUTDIR}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        dtype=dtype,                 # 新API名：dtype（非 torch_dtype）
        low_cpu_mem_usage=True,
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    # 极短 prompt，生成 1 个 token
    prompt = "你好"
    enc = tokenizer(prompt, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}

    gen_kwargs = dict(
        max_new_tokens=256,           # 只跑 3 个 token
        do_sample=False,            # 确定性
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    )

    # --- Profiler：记录内存 + 形状 + 调用栈（不使用 on_trace_ready，避免重复保存冲突）---
    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,    # ✅ 记录内存
        with_stack=True,
        with_modules=True,
    ) as prof:
        with torch.no_grad():
            with record_function("generate_one_token"):
                out = model.generate(**enc, **gen_kwargs)
            if device == "cuda":
                torch.cuda.synchronize()
        # 标记一步完成（可选，但有助于内存时间线分段）
        prof.step()

    # --- 导出报告 ---
    trace_path = os.path.join(OUTDIR, "trace_one_token.json")
    mem_html = os.path.join(OUTDIR, "memory_timeline.html")

    # Chrome trace（在 chrome://tracing 打开）
    prof.export_chrome_trace(trace_path)
    print(f"[Saved] Chrome trace: {trace_path}")

    # 内存时间线（用浏览器打开）
    # device 也可用 "cuda" 或具体 "cuda:0"
    prof.export_memory_timeline(mem_html, device="cuda:0" if device == "cuda" else "cpu")
    print(f"[Saved] Memory timeline: {mem_html}")

    # 打印关键耗时汇总（按 CUDA 时间 or CPU 时间排序）
    sort_key = "cuda_time_total" if device == "cuda" else "self_cpu_time_total"
    print("\n=== Key Averages ===")
    print(prof.key_averages().table(sort_by=sort_key, row_limit=30))

    # 展示结果文本（包含 1 个新增 token）
    txt = tokenizer.decode(out[0], skip_special_tokens=True)
    print("\n=== Output ===\n", txt)

if __name__ == "__main__":
    main()
