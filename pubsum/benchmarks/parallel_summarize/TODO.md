# Parallel Summarization Benchmark TODO

1. Get/set system CPU/GPU count automatically
2. Add GPU/CPU count and abstracts per job to results
3. Should we be doing replicates or collecting individual job run times?

## GOTCHYAs

Saw the following warning for the first time after multiple repeated runs. Don't think anything was changed in the benchmark code since the last successful run except the number of abstracts to summarize:

```
Job 11: starting.

Starting benchmark with 1 concurrent jobs and 120 abstracts per job using CPU physical cores only huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
Job 0: starting.
```

htop shows no CPU usage at all, but we should be using 10 threads. Also, GPU jobs from the same run appeared to work just fine.

UPDATE: Proofread a little and made some small edits, don't think I really 'fixed' anything, but now it seems to be working. Really not sure what I did.
