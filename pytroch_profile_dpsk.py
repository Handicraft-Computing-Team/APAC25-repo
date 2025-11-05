import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler
from contextlib import nullcontext

def load_local_model(model_dir: str, device: str, dtype):
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    return tokenizer, model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="/home/haibin/apac25/dpsk8B")
    parser.add_argument("--outdir", type=str, default="./prof_out")  # TensorBoard 日志目录
    parser.add_argument("--prompt", type=str,
                        default="请用要点说明Transformer的注意力是如何工作的。给出直观解释。")
    parser.add_argument("--max_new_tokens", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--wait", type=int, default=2)      # Profiler 调度：等待步数
    parser.add_argument("--warmup", type=int, default=2)    # 预热步数（记录但不导出 trace）
    parser.add_argument("--active", type=int, default=4)    # 真实记录步数（导出 trace）
    parser.add_argument("--steps", type=int, default=8)     # 总步数 = wait+warmup+active（可更大）
    parser.add_argument("--use_compile", action="store_true",
                        help="在 PyTorch 2.x 上对模型进行 torch.compile 加速（可能更快，trace更干净）")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    print(f"[Info] Loading local model from: {args.model_dir}")
    print(f"[Info] Device={device}, dtype={dtype}, outdir={args.outdir}")
    tokenizer, model = load_local_model(args.model_dir, device, dtype)

    # 可选：torch.compile（需要 PyTorch 2.1+）
    compile_ctx = nullcontext()
    # if args.use_compile and hasattr(torch, "compile"):
    #     print("[Info] Using torch.compile(...)")
    #     model = torch.compile(model)
    # else:
    #     print("[Info] Not using torch.compile")

    # 预先编码一次，避免把 tokenizer 时间记入模型性能
    enc = tokenizer(args.prompt, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    )

    # 为了让每 step 都做同样工作，复制一份输入（不必每步重新tokenize）
    def run_once():
        with torch.no_grad():
            out = model.generate(**enc, **gen_kwargs)
            max_new_tokens=3,
            # 同步确保 CUDA 时间被记录完整
            if device == "cuda":
                torch.cuda.synchronize()
        return out

    # prof_sched = schedule(wait=args.wait, warmup=args.warmup, active=args.active, repeat=1)
    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)

    print(f"[Info] Profiler schedule: wait={args.wait}, warmup={args.warmup}, active={args.active}")
    print(f"[Info] Total steps to run: {args.steps}")

    with profile(
        activities=activities,
        # schedule=prof_sched,
        on_trace_ready=tensorboard_trace_handler(args.outdir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for step in range(args.steps):
            _ = run_once()
            prof.step()

    prof.export_chrome_trace(os.path.join(args.outdir, "trace.json"))

    # 打印关键耗时汇总（按 CUDA 时间排序）
    print("\n================= Key Averages (by CUDA time) =================")
    try:
        print(prof.key_averages().table(
            sort_by="cuda_time_total" if device == "cuda" else "self_cpu_time_total",
            row_limit=30
        ))
    except Exception:
        print(prof.key_averages().table(row_limit=30))

    print(f"\n[Done] Traces are saved under: {args.outdir}")
    print("Open with TensorBoard:\n  tensorboard --logdir ./prof_out   (then open the shown URL)")
    print("In TensorBoard, go to: ‘Profile’ tab -> ‘trace_viewer’ to inspect CUDA/CPU timeline.")

if __name__ == "__main__":
    main()
