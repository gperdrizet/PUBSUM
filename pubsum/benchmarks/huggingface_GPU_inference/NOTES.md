# Notes

1. FlashAttention-2 not implemented by model.
2. Attempting to load model with 8 bit quantization gives the following error:

```
Error named symbol not found at line 529 in file /mmfs1/gscratch/zlab/timdettmers/git/bitsandbytes/csrc/ops.cu
```

Also, sometimes getting the following warning...

```
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
Error named symbol not found at line 74 in file /mmfs1/gscratch/zlab/timdettmers/git/bitsandbytes/csrc/ops.cu
```

A little strange since we are not using multiprocessing for this benchmark - it should be a single job on a single GPU. Did sporadically see this warning when working on the parallel summarization benchmark too, never figured out what it was because it was not reliably reproducible.

Fixed by setting TOKENIZERS_PARALLELISM environment variable to true before run start.