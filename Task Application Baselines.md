

# Task Application Baselines

## NWChem

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

## DeepSeek SGLang

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

