#!/usr/bin/env python3
import optuna, math, os
import matplotlib.pyplot as plt
import pandas as pd

# === 搜索空间 ===
stack_vals  = list(range(4000, 16001, 500))
heap_vals   = list(range(50, 401, 50))
global_vals = list(range(4000, 32001, 1000))
mpi_strats  = ["ARMCI-MPI", "MPI-PR", "MPI-TS"]

# === 初始点（你的要求）===
initial_point = {"stack_mb": 8000, "heap_mb": 100, "global_mb": 8000, "mpi_strategy": "MPI-PR"}

# === 创建Study ===
study = optuna.create_study(
    study_name="manual_nwchem_visual",
    storage="sqlite:///manual_tune.db",
    direction="minimize",
    sampler=optuna.samplers.GridSampler({
        "stack_mb": stack_vals,
        "heap_mb": heap_vals,
        "global_mb": global_vals,
        "mpi_strategy": mpi_strats,
    }),
    load_if_exists=True,
)

print(f"共 {len(stack_vals)*len(heap_vals)*len(global_vals)*len(mpi_strats)} 个组合。")
print("提示：Ctrl+C 可随时中断，重新运行会从未完成的组合继续。\n")

# === 绘图函数 ===
def plot_progress(df: pd.DataFrame):
    plt.figure(figsize=(8,6))
    plt.title("NWChem 参数调优可视化 (分数越小越好)", fontsize=12, fontweight="bold")
    plt.scatter(df["trial_id"], df["value"], c="blue", label="输入分数")
    plt.xlabel("Trial #")
    plt.ylabel("Score (如 wall-time)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("tune_progress.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ 已更新可视化: tune_progress.png")

def objective(trial):
    s = trial.suggest_categorical("stack_mb", stack_vals)
    h = trial.suggest_categorical("heap_mb", heap_vals)
    g = trial.suggest_categorical("global_mb", global_vals)
    m = trial.suggest_categorical("mpi_strategy", mpi_strats)

    print(f"\n=== 当前组合 ===")
    print(f"stack={s}MB, heap={h}MB, global={g}MB, MPI={m}")
    val = None
    while val is None:
        try:
            inp = input("请输入此组合的测量结果（数值越小越好）: ").strip()
            val = float(inp)
        except ValueError:
            print("❌ 请输入数字，例如 123.45")
    trial.set_user_attr("manual_input", True)
    return val

# === 优先测试初始点 ===
found = any(t.params == initial_point for t in study.trials)
if not found:
    print("\n>>> 先测试初始点: ", initial_point)
    y = None
    while y is None:
        try:
            y = float(input("请输入初始点的测量结果: "))
        except ValueError:
            print("❌ 请输入数字")

    manual_trial = optuna.trial.create_trial(
        params=initial_point,
        distributions={
            "stack_mb": optuna.distributions.CategoricalDistribution(stack_vals),
            "heap_mb": optuna.distributions.CategoricalDistribution(heap_vals),
            "global_mb": optuna.distributions.CategoricalDistribution(global_vals),
            "mpi_strategy": optuna.distributions.CategoricalDistribution(mpi_strats),
        },
        value=y,
        user_attrs={"manual_input": True, "is_initial": True}
    )
    study.add_trial(manual_trial)
    print("✅ 已记录初始点结果。\n")


# === 主循环 ===
try:
    study.optimize(objective, n_trials=None, gc_after_trial=True)
except KeyboardInterrupt:
    print("\n停止调优。")

# === 汇总 + 绘图 ===
valid = [t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE and math.isfinite(t.value)]
if valid:
    df = pd.DataFrame([
        {"trial_id": t.number, **t.params, "value": t.value} for t in valid
    ])
    df.to_csv("tune_results.csv", index=False)
    plot_progress(df)

    best = min(valid, key=lambda t: t.value)
    print("\n=== 当前最优结果 ===")
    print(f"最小值: {best.value}")
    print("参数:", best.params)
else:
    print("⚠️ 暂无有效 trial。")
