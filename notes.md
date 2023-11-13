# Progress update

Everything needed to start summarizing abstracts is in place. We have 3.68 million abstracts in an SQL library and ready to be served via an in-house postgreSQL server. But, my guess is that it is going to take a very long time to summarize them all. Let's do a benchmark run with a small sample and see how we are doing in terms of throughput.

To my current knowledge level - we are doing the simplest thing possible:

1. Open connection to postgreSQL and create a cursor for writing.
2. Check existence of and create or empty a target table for abstract summaries as appropriate.
3. Create read cursor and load n abstracts.
4. Load summarization model/tokenizer.
5. Loop on abstracts, creating abstract summaries and inserting them into the target table.
6. Time the whole loop and total time spent summarizing and inserting, calculate time spent loading/looping as difference.

## Initial summarization rate benchmark results

Here are the timing results from summarizing 10 abstracts:

```
Total time: 137.3
Summarization time: 136.3
Insert time: 0.994
Loading time: 0.0006

Mean summarization time: 13.6
Mean insert time: 0.099
Mean loading time: 0.0001
```

Not great, at least not if the goal is to summarize all of the abstracts. Mean time per article is 13.7-ish seconds - that works out to about 585 days to summarize all of the abstracts. Not surprisingly, the bulk of the time is spent in summarization - 99.3%. On one hand, this is good news, our in-house SQL server is far from the limiting factor. If we summarized the whole dataset right now, only about 5 of the 585 days that it would take would be spent on SQL IO. 

Let's see if we can improve the model's rate of summarization a bit. Here are a few ideas:

1. Use the GPUs - we have 2 NVIDIA K80s for a total of 4, GK210 chips each with 12 GB of GDDR5.
2. Run multiple summarizers in parallel - it should, in theory, be possible to run more than on job on each GPU. Tensorflow can do this (assuming the model and data fit). Not sure about torch and huggingface transformers.
3. Other optimizations: huggingface has a good article on [GPU inference](https://huggingface.co/docs/transformers/perf_infer_gpu_one) that it would probably be worth working through, even if just for general knowledge's sake. Possible strategies include: flashattention-2, BetterTransformer, bitsandbytes, Optimum and 4 or 8 bit model quantization.

In order to experiment with all of that well, I think the first thing we should do is make our throwaway summarizer script into a good benchmarking harness.
