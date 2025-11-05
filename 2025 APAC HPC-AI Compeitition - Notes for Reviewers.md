# Notes for Reviewers - 2025 APAC HPC-AI Compeitition

## What this note is for

- Explain the nature of the two tasks (HPC NWChem and AI DeepSeek SGLang offline inference), the shared clusters and constraints, and how these affect scoring.
- Provide per-cluster baselines so reviewers know what “out-of-the-box” looks like.
- Avoid rewarding brute-force parameter sweeps that just burn cycles without insight.

## Shared competition context

- We provide three reference clusters for teams to run and tune. This spreads the load so no single co-organizer bears all compute pressure.
- Consequence: raw performance numbers across clusters are not directly comparable(e.g., Intel Sapphire Rapids on Gadi vs. AMD EPYC Milan on Aspire2A).
- Therefore, scoring emphasizes the approach, reasoning, and evidence over absolute cross-cluster speed.

## Scoring emphasis (what we value)

- Approach and reasoning: chosen application/software stack/system-level optimizations, why they fit the workload, and evidence they work.
- Clarity and understanding: can the team explain what improved, why it improved, and trade-offs?
- Reproducible, verifiable evidence on the clusters they used.
- We do not reward blind, massive parameter sweeps that consume huge resources and “find a best number” without understanding.

## What good looks like (HPC NWChem)

- **Metrics definition**: Compare runs using Wall Time and Wall CPU Time reported by NWChem; lower is better. We will verify that energies match within the expected tolerance during offline result checks.
- Shows methodical single-node and ≤4-node tuning.
- Justifies compiler/MPI/BLAS choices; shows pinning, affinity, I/O, memory, and any other application-level considerations.
- Clearly provides minimal effective options/flags/configs, and reproducible command lines.
- Includes performance profiling: identify hotspots and bottlenecks (CPU, memory bandwidth, NUMA, I/O, MPI), with before/after evidence (e.g., perf, VTune, mpiP, HPCToolkit, LIKWID).
- Explains scaling factors of this NWChem workload (for w=12) and adapts strategy accordingly.

## What good looks like (AI DeepSeek SGLang Inference)

- **Metric definition**: Total token throughput refers to input+output tokens per second aggregated over all concurrent requests; higher is better.
- Focus on optimization choices, engineering rigor, and reproducibility rather than a single best number from a giant sweep.
- Clear ablations or targeted experiments that justify the improvements.
- Includes performance profiling: trace/measure throughput–latency, kernel-level hotspots (compute vs. input pipeline), memory/PCIe/NVLink utilization, and correctness–speed trade-offs; show profiler screenshots/log extracts and how they informed changes.

## Baselines you can rely on

- For each task and each reference cluster we provide a baseline: the simplest “supercomputer beginner” run.
- Example (HPC/NWChem): site-provided NWChem via environment modules on shared storage (admin precompiled module on Gadi/NSCC). Teams start from here, then:
  - Step 1: single-node tuning (choose compilers/MPI/BLAS, flags, I/O, process/thread pinning, etc.)
  - Step 2: best performance within 4 nodes
- We kindly ask the judges to review the gains over the out-of-the-box baseline, and whether the path is technically sound and clearly explained.

## Important constraints behind the NWChem task

- Task: a reduced version of “c. Density functional theory” from the https://hpcadvisorycouncil.atlassian.net/wiki/spaces/HPCWORKS/pages/2799534081/Getting+Started+with+NWChem+for+ISC22+SCC#Tasks-and-Submissions (w set to 12).
- On modern CPUs with 100+ physical cores per node, this workload is in practice poorly scalable at larger sizes.
- Why limit to w=12 and ≤4 nodes?
  - As size grows, not only compute iteration time but also preparation and pre-processing across the cluster grow very fast (nonlinear). Each run becomes very expensive in wall time.
  - With many student teams sharing resources, the problem must remain tractable.
- Reality check: some stacks may not build/run cleanly with NWChem (e.g., AOCC, newer MKL, newer GCC, certain combos). The upstream community support may be limited. Some teams may report approach/lessons-learned without fully working binaries.

## How to read results across clusters

- Do not directly compare raw timings between clusters with different CPU generations/architectures/interconnects.
- Compare each team against:
  - The baseline on the same cluster, and
  - The team’s own earlier runs on that cluster (their improvement curve),
  - The coherence of their method and analysis.

## Scoring guardrails

- Evidence-based: Score verifiable results demonstrated.
- Do not penalize for presentation language/accent; judge the technical substance.

## In a word

- Keep scoring smooth and evidence-based. Because clusters differ and resource are shared, please prioritize method, understanding, and justified improvements over absolute speed across clusters.



## Quick references for reviewers

- NWChem getting started (ISC22 SCC): https://hpcadvisorycouncil.atlassian.net/wiki/spaces/HPCWORKS/pages/2799534081/Getting+Started+with+NWChem+for+ISC22+SCC
- About SGLang bench_offline_throughput: https://docs.sglang.ai/developer_guide/benchmark_and_profiling.html
- About SGLang offline benchmark: https://lmsys.org/blog/2024-07-25-sglang-llama3/#benchmark-setup
- HPC-AI Competition tasks rules and application notes: https://github.com/hpcac/2025-APAC-HPC-AI/tree/main



# Task Application Baselines

## NWChem (Wall CPU Time and Wall Time as metrics)

### ASPIRE2A

#### Benchmark timing result

```
job.nwchem.w12.nodes1.ncpus128.stdout: Total times  cpu:       99.9s     wall:      100.9s
```

#### Energy results

```
job.nwchem.w12.nodes1.ncpus128.stdout:         Total DFT energy =     -917.738244976750
job.nwchem.w12.nodes1.ncpus128.stdout: Nuclear repulsion energy =     1015.516447774855
```

### Gadi

#### Benchmark timing result

```
job.nwchem.w12.nodes1.ncpus104.stdout: Total times  cpu:      237.3s     wall:      240.9s
```

#### Energy results

```
job.nwchem.w12.nodes1.ncpus104.stdout:         Total DFT energy =     -917.738244977091
job.nwchem.w12.nodes1.ncpus104.stdout: Nuclear repulsion energy =     1015.516447774856
```

## DeepSeek SGLang (Total token throughput as metrics)

### ASPIRE2P

#### Benchmark throughput result

```
sglang.sh.o74993:[1,0]<stdout>:====== Offline Throughput Benchmark Result =======
sglang.sh.o74993-[1,0]<stdout>:Backend:                                 engine    
sglang.sh.o74993-[1,0]<stdout>:Successful requests:                     2000      
sglang.sh.o74993-[1,0]<stdout>:Benchmark duration (s):                  173.88    
sglang.sh.o74993-[1,0]<stdout>:Total input tokens:                      626729    
sglang.sh.o74993-[1,0]<stdout>:Total generated tokens:                  388685    
sglang.sh.o74993-[1,0]<stdout>:Last generation throughput (tok/s):      34.25     
sglang.sh.o74993-[1,0]<stdout>:Request throughput (req/s):              11.50     
sglang.sh.o74993-[1,0]<stdout>:Input token throughput (tok/s):          3604.32   
sglang.sh.o74993-[1,0]<stdout>:Output token throughput (tok/s):         2235.33   
sglang.sh.o74993-[1,0]<stdout>:Total token throughput (tok/s):          5839.65   
sglang.sh.o74993-[1,0]<stdout>:==================================================
```

### SMC

#### Benchmark throughput result

```
slurm-824.out:====== Offline Throughput Benchmark Result =======
slurm-824.out-Backend:                                 engine    
slurm-824.out-Successful requests:                     2000      
slurm-824.out-Benchmark duration (s):                  85.41     
slurm-824.out-Total input tokens:                      626729    
slurm-824.out-Total generated tokens:                  388685    
slurm-824.out-Last generation throughput (tok/s):      71.39     
slurm-824.out-Request throughput (req/s):              23.42     
slurm-824.out-Input token throughput (tok/s):          7338.13   
slurm-824.out-Output token throughput (tok/s):         4550.97   
slurm-824.out-Total token throughput (tok/s):          11889.10  
slurm-824.out-==================================================
```

