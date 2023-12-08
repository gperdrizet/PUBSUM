# Notes

After completing 3 jobs per GPU with batch size of 4 (resulting OOM for every replicate), recived the following error on the first 3 jobs per GPU with batch size 8 job:

```
Traceback (most recent call last):
  File "run_benchmarks.py", line 261, in <module>
    parallel_batched.benchmark(
  File "/home/siderealyear/arkk/huggingface_transformers/pubsum/benchmarks/parallel_batched_summarization/benchmark.py", line 143, in benchmark
    result = [async_result.get() for async_result in async_results]
  File "/home/siderealyear/arkk/huggingface_transformers/pubsum/benchmarks/parallel_batched_summarization/benchmark.py", line 143, in <listcomp>
    result = [async_result.get() for async_result in async_results]
  File "/usr/lib/python3.8/multiprocessing/pool.py", line 771, in get
    raise self._value
RuntimeError: CUDA error: CUBLAS_STATUS_NOT_INITIALIZED when calling `cublasCreate(handle)`
```