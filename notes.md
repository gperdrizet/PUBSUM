# Progress update

Everything needed to start summarizing abstracts is in place. We have 3.68 million abstract though, my guess it that it is going to take a very long time to summarize them all. Let's do a benchmark run with a small sample and see how we are doing in terms of throughput.

To my current knowledge level - we are doing the simplest thing possible:

1. Open connection to postgreSQL and create cursor.
2. Check existence of and create or empty target table for abstract summaries as appropriate.
3. Create read cursor and load n abstracts.
4. Load summarization model/tokenizer.
5. Loop on abstracts, creating summary and inserting into target table
6. Time whole loop and total time spent summarizing and inserting, calculate time spent loading.
