
# SUSTech APAC HPC-AI 2025 Student Competition Technique Report

APAC HPC-AI 2025 Student Competition

Team:
SUSTech Scout Regiment

Captain:
Haibin Lai

Email:
laihb2022@mail.sustech.edu.cn


## Introduction

The **SUSTech Scout Regiment** is a passionate and multidisciplinary team from the **Southern University of Science and Technology (SUSTech)**. Our members come from different academic years and research backgrounds, united by a shared enthusiasm for **high-performance computing** and **scientific software optimization**. We aim to tackle real-world supercomputing challenges while learning from hands-on system-level experiments and performance tuning.

![image.png](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/FinalYear/20251105103551010.png)


In this year’s competition, our team utilized **two distinct high-performance computing environments** to evaluate and optimize our workloads.

For the **CPU track**, we conducted experiments on the **Qiming 2.0 cluster** at SUSTech, which is equipped with **Intel® Xeon® Gold 6138 CPUs** and a **Lustre parallel file system**. This environment closely resembles the **Gadi** and **NSCC** reference platforms provided by the competition organizers. We executed **NWChem workloads** on up to **four nodes (160 CPU cores)** to evaluate scalability, communication efficiency, and memory utilization, while applying multi-level optimization techniques to enhance performance.

![image.png](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/FinalYear/20251105103605573.png)


For the **GPU track**, our **DeepSeek-R1** experiments were performed on the **organizers’ H200 reference cluster**. We are deeply grateful for the opportunity to access this advanced infrastructure, which allowed our software-level optimizations and fine-grained kernel tuning to fully demonstrate their performance potential.

![image.png](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/FinalYear/20251105103619068.png)


## NWChem

### Overview

**NWChem** is a scalable, open-source computational chemistry package designed for large-scale simulations on supercomputers. It supports a wide range of scientific models, including **quantum chemistry** and **molecular dynamics**.  
Among its core components, the **Self-Consistent Field (SCF)** and **Density Functional Theory (DFT)** modules are particularly representative.  
SCF workloads iteratively solve the electronic wavefunction, while DFT workloads extend this with **grid-based numerical integrations**.  
Both workloads are **communication-intensive**, relying heavily on **MPI** and **Global Arrays (GA)** to perform distributed computation across nodes.

![image.png](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/FinalYear/20251105103958570.png)


### Workload Description

In NWChem, data is organized in **Global Arrays**, which provide a shared-memory abstraction on top of MPI.  
During each SCF or DFT iteration, processes **fetch sub-blocks of distributed arrays**, perform local computations, and then **write the results back** to the global data structure.  
This design simplifies programming but introduces synchronization and communication overhead.

![image.png](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/FinalYear/20251105103856810.png)


For our experiments, we ran the **baseline NWChem** build using **Intel MPI** and **OpenBLAS**.  
We evaluated performance based on **wall-clock time**, ensuring that all outputs were scientifically correct.  
However, we observed **limited scalability beyond four nodes**, indicating potential bottlenecks in communication and computation.

![image.png](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/FinalYear/20251105103915033.png)


### Performance Analysis

We used **Intel VTune Profiler** to analyze runtime behavior and identify bottlenecks.  
The results revealed that approximately **28% of total execution time** was spent in **communication**, primarily due to **MPI_Barrier** and **MPI_Iprobe** operations.  
This suggests **load imbalance** across processes, where some ranks finish computation earlier and wait for synchronization.

The second major bottleneck lies in **computation efficiency**.  
Many of NWChem’s core operators are **memory-bound**, meaning their performance is constrained by memory bandwidth rather than compute throughput.  
As a result, optimization opportunities mainly come from **compiler-level tuning**, **loop transformations**, and **I/O improvements** rather than algorithmic restructuring.

Finally, we identified significant time spent in **BLAS routines**, especially **DGEMM (Double-precision General Matrix Multiplication)**.  
These are computationally heavy but can be improved through **optimized BLAS libraries**, **multi-threading**, or **NUMA-aware scheduling**.

![image.png](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/FinalYear/20251105103930198.png)


### MPI Profiling Insights

Using **IPM (Integrated Performance Monitoring)**, we further examined the MPI communication pattern.  
The profiling results show that a large proportion of processes frequently invoke **MPI_Iprobe**, indicating they are **polling for incoming data** instead of performing useful computation.  
This behavior confirms our earlier observation: the system suffers from **communication waiting and load imbalance**, which severely limits parallel efficiency.

![image.png](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/FinalYear/20251105103942433.png)




### Optimization Strategies

Based on the performance bottlenecks identified through **VTune** and **IPM** profiling, we proposed and implemented **five key optimizations** targeting both computation and communication inefficiencies.

![image.png](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/FinalYear/20251105103958570.png)

#### 1. Compiler and BLAS Optimization

To improve computational efficiency, we experimented with multiple compiler and BLAS combinations.  
We compared **Intel OneAPI (ICC + MKL)** and **GCC + OpenBLAS**, observing that MKL generally provides better memory alignment and vectorization performance for dense matrix operations such as **DGEMM**.  

![image.png](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/FinalYear/20251105105951460.png)


By enabling advanced compiler flags (e.g., `-O3`,  and `-fopenmp`), we improved the efficiency of memory-bound operators within SCF and DFT kernels.

![image.png](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/FinalYear/20251105105946517.png)



#### 2. NUMA + OpenMP + MPI Hybrid Parallelism

To better exploit on-node parallelism, we introduced **hybrid OpenMP + MPI parallelization**.  
Each node runs fewer MPI ranks, with OpenMP threads handling intra-node computations.  
This reduces inter-rank communication and makes better use of shared caches and NUMA domains.  
We observed that OpenMP+MPI is not allowed in SCF and DFT workload, but we can still do the NUMA binding.

![image.png](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/FinalYear/20251105104151061.png)


#### 3. Communication and Load-Balancing Optimization

Communication imbalance—evident from excessive time in **MPI_Barrier** and **MPI_Iprobe**—was addressed by experimenting with **different MPI implementations and configurations**.  

We tested **Intel MPI**, **hpcx**, and **OpenMPI**, tuning runtime parameters  and collective algorithms. 
![image.png](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/FinalYear/20251105105933725.png)
 

Moreover, we explored **MPI process remapping** to improve affinity and reduce latency.  
For the most communication-heavy phases, we evaluated **MPI-PR (Persistent Request)** strategies to overlap communication with computation, which reduced idle waiting time.

![image.png](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/FinalYear/20251105105927083.png)


![image.png](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/FinalYear/20251105104337954.png)


#### 4. Memory Allocation

Memory locality significantly impacts the performance of NWChem’s Global Arrays.  
We applied bayesian Optimization on Memory allocation.
This tuning effectively reduced remote memory access and improved memory bandwidth utilization across nodes.

![image.png](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/FinalYear/20251105104349441.png)

![image.png](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/FinalYear/20251105104407437.png)


![image.png](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/FinalYear/20251105105918026.png)


---

### Summary of Improvements

After applying these optimizations, we observed measurable performance gains across SCF and DFT workloads.  
The  optimized compiler + BLAS configuration delivered faster computation kernels.  
Collectively, these improvements led to **better scalability and reduced wall time on up to four nodes**, demonstrating the importance of **co-designing communication, computation, and memory strategies** for scientific HPC applications.

![image.png](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/FinalYear/20251105104456902.png)

![image.png](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/FinalYear/20251105104513649.png)

![image.png](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/FinalYear/20251105105534796.png)

## DeepSeek-R1 Optimization

![image.png](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/FinalYear/20251105104929006.png)

![image.png](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/FinalYear/20251105104744861.png)

DeepSeek-R1 (671B) achieves strong performance across various reasoning and language tasks. 
Its MLA architecture enables more efficient use of attention, while the MoE design activates only 37 B parameters per inference.

![image.png](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/FinalYear/20251105105910004.png)



SGLang supports Tensor Parallelism (TP), Pipeline Parallelism (PP), and Data Parallelism (DP), enabling efficient large-scale distributed inference across multiple GPUs and nodes.
It also integrates advanced features such as speculative decoding and Radix Attention, which significantly improve decoding throughput and reduce latency.
Through studying and experimenting with SGLang, we gained a deep understanding of its scheduling and parallel strategies. In our benchmarks, it demonstrated strong scalability and excellent performance efficiency on modern GPU clusters.

![image.png](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/FinalYear/20251105104810325.png)


### Baseline Setup

For the **GPU track**, we conducted experiments on the **organizers’ reference H200 cluster**, using a **two-node, 16-GPU configuration** (8 × H200 per node, connected with NVLink + InfiniBand).  
We evaluated multiple **parallelism strategies**, including:

- **Tensor Parallelism (TP)** – splitting matrix multiplications across GPUs,
    
- **Pipeline Parallelism (PP)** – dividing layers across nodes to balance memory usage, and
    
- **Data Parallelism (DP)** – replicating models to process multiple requests concurrently.
    

Our baseline employed the **SGLang engine** running with the **FA3 backend**, which served as a stable starting point for functional and performance correctness.

![image.png](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/FinalYear/20251105104838217.png)


### Bottleneck Profiling

We used **NVIDIA Nsight Systems (nsys)** to profile both prefill and decode stages.  
Profiling results indicated that the main performance bottleneck came from **communication overhead** rather than computation.  
Specifically, NCCL traces showed numerous small but frequent data exchanges between GPUs, leading to **high communication latency** and **low link utilization**.  
In multi-node cases (PP > 1 or DP > 1), synchronization between pipeline stages further aggravated the delay, dominating overall iteration time.

![image.png](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/FinalYear/20251105104846518.png)


![image.png](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/FinalYear/20251105104904867.png)
![image.png](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/FinalYear/20251105104918511.png)

![image.png](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/FinalYear/20251105104929006.png)



### 1. Parallelization Strategy
**

For TP=8 and PP=2, all 8 H200 GPUs within a node are used for TP, while PP spans nodes — minimizing inter-node communication.

  

DP = 2 enables dual-node concurrency with minimal communication, doubling throughput while keeping TP/PP balanced inside each node.

**

![image.png](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/FinalYear/20251105105004525.png)


1. **Parallelism Strategy Exploration**  
    We systematically tested combinations of (TP, PP, DP)  to analyze the trade-off between inter- and intra-node communication.  
    The configuration **TP = 8, PP = 2** achieved the best balance — using intra-node tensor parallelism to minimize cross-node data transfer while leveraging pipeline parallelism for memory scalability.
![image.png](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/FinalYear/20251105105825021.png)


![image.png](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/FinalYear/20251105105831994.png)


### 2. **NCCL Communication**

We tuned NCCL parameters and communication strategies to better suit the small-message, high-frequency workload pattern.  
Tree-based algorithms replaced ring algorithms for latency-sensitive collectives, and the **LL128 protocol** was adopted for small transfers.  

  ![image.png](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/FinalYear/20251105105132840.png)


## 3. backend

**

FA3, the default SGLang kernel, already achieves the best overall throughput among all backend implementations

**

![image.png](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/FinalYear/20251105105205145.png)

During the competition, we initially encountered difficulties running **FlashInfer** within the SGLang framework. Profiling showed communication stalls that prevented correct execution under certain (TP, PP, DP) configurations. After carefully reproducing and analyzing the issue, we reported our findings to the **SGLang official development team** through a detailed GitHub issue, as also suggested by the judges.

Subsequent investigation by the SGLang developers confirmed that the problem originated from an **in-development communication bug** in the FlashInfer backend. The issue has since been **officially fixed** in the latest SGLang release. We re-tested the workload on the **two-node, 16-GPU H200 cluster**, and **FlashInfer now runs successfully** with stable throughput and correct outputs.

issue https://github.com/sgl-project/sglang/issues/12402

This collaboration not only helped us complete our benchmark successfully but also contributed to improving the SGLang ecosystem. We are grateful to the organizers and the SGLang team for their timely support.

![13b00e43df060a372e1d21d7eff4a69f.png](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/FinalYear/20251105105313742.png)


## 3. **Speculative Decoding**

**

Speculative decoding accelerates language model inference by using a smaller draft model to predict several tokens ahead, which the main model then verifies in parallel. To further improve throughput in the decode stage, we implemented **speculative decoding**, which generates multiple tokens concurrently using a lightweight draft model.

**

![image.png](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/FinalYear/20251105105841824.png)


![image.png](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/FinalYear/20251105105847787.png)



But Speculative Decoding is not allowed with DP Parallelism!

Abort since not as much gain as DP

### Summary of Results

After applying these optimizations, we observed:

- **Reduced NCCL latency** and improved GPU utilization,
    
- **Better overlap** between compute and communication phases,
    
- **Higher decoding throughput**, especially when speculative decoding was enabled, and
    
- **Improved stability** across different backend configurations.
    

Overall, our optimized setup achieved **significant end-to-end speedups** over the baseline, demonstrating that **communication-aware parallelism** and **kernel-level backend tuning** are essential for scaling LLM inference efficiently on multi-GPU, multi-node systems.

![image.png](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/FinalYear/20251105105447479.png)

![image.png](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/FinalYear/20251105105457277.png)
