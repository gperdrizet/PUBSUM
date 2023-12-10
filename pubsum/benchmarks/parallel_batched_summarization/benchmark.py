import os
import gc
from typing import List, Tuple
import pandas as pd
import psycopg2
import time
import torch
import itertools
import multiprocessing as mp
from .. import helper_functions as helper_funcs
import transformers

def benchmark(
    resume: str,
    results_dir: str,
    replicates: int,
    batches: int,
    batch_sizes: List[int],
    workers: List[int],
    gpus: List[str],
    quantization_strategies: List[str],
    db_name: str, 
    user: str, 
    passwd: str, 
    host: str
) -> bool:
    
    print(f'\nRunning data parallel, batched summarization benchmark. Resume = {resume}.\n')

    # Set list of keys for the data we want to collect
    collection_vars = [
        'abstracts',
        'batches',
        'replicate',
        'batch size',
        'workers',
        'workers per GPU',
        'quantization',
        'summarization time (sec.)',
        'summarization rate (abstracts/sec.)'
    ]

    # Subset of collection vars which are sufficient to uniquely identify each run
    unique_collection_vars = [
        'quantization',
        'workers',
        'batch size',
        'replicate'
    ]

    # Handel resume request by reading or emptying old data, as appropriate
    completed_runs = helper_funcs.resume_run(
        resume=resume, 
        results_dir=results_dir,
        collection_vars=collection_vars,
        unique_collection_vars=unique_collection_vars
    )

    # Construct parameter sets
    replicate_numbers = list(range(1, replicates + 1))

    parameter_sets = itertools.product(
        quantization_strategies,
        workers,
        batch_sizes,
        replicate_numbers
    )

    # Loop on parameter sets to run jobs
    for parameter_set in parameter_sets:

        # Check if we have already completed this parameter set
        if parameter_set not in completed_runs:

            # Unpack parameters from set
            quantization, workers, batch_size, replicate = parameter_set

            # Calculate total abstracts needed for job
            num_abstracts = batches * batch_size

            print(f'\nParallel batched summarization:\n')
            print(f' Replicate: {replicate}')
            print(f' Model quantization: {quantization}')
            print(f' Batch size: {batch_size}')
            print(f' Batches: {batches}')
            print(f' Workers: {workers}\n')

            # Instantiate results object for this run
            results = helper_funcs.Results(
                results_dir=results_dir,
                collection_vars=collection_vars
            )

            # Collect data for run parameters
            results.data['abstracts'].append(num_abstracts)
            results.data['replicate'].append(replicate)
            results.data['batches'].append(batches)
            results.data['batch size'].append(batch_size)
            results.data['workers'].append(workers)
            results.data['workers per GPU'].append(workers // len(gpus))
            results.data['quantization'].append(quantization)

            # start a counter to pick GPUs for jobs
            gpu_index = 0

            # Instantiate pool with one member for each job we need to run
            pool = mp.Pool(
                processes = workers,
                maxtasksperchild = 1
            )

            # Start timer
            start = time.time()

            # Collector for returns from workers
            async_results = []

            # Loop on jobs for this run
            for i in list(range(0, workers)):

                # Pick GPU
                gpu = gpus[gpu_index]

                async_results.append(
                    pool.apply_async(start_job,
                        args = (
                            i,
                            db_name,
                            user,
                            passwd,
                            host,
                            num_abstracts,
                            batches,
                            batch_size,
                            gpu_index,
                            gpu,
                            quantization
                        )
                    )
                )

                # Increment GPU index - we have four GPUs, so when the index gets 
                # to 3 (0 anchored), reset it back to 0, otherwise increment it.
                if gpu_index == 3:
                    gpu_index = 0
                
                else:
                    gpu_index += 1

            # Clean up
            pool.close()
            pool.join()

            # Stop the timer
            dT = time.time() - start

            # Get the results
            result = [async_result.get() for async_result in async_results]
            print(f'\n Workers succeeded: {result}')

            # Collect and save data, if we returned an OOM error, mark it in the results
            if False not in result:
                results.data['summarization time (sec.)'].append(dT)
                results.data['summarization rate (abstracts/sec.)'].append((batches * batch_size * workers)/dT)

            else:
                results.data['summarization time (sec.)'].append('OOM')
                results.data['summarization rate (abstracts/sec.)'].append('OOM')

            results.save_result()

    return True


def start_job(
    i: int, 
    db_name: str, 
    user: str, 
    passwd: str, 
    host: str,
    num_abstracts: int,
    batches: int, 
    batch_size: int, 
    gpu_index: int, 
    gpu: str, 
    quantization: str
) -> bool:
    
    print(f' Job {i}: starting on {gpu}.')

    try:
        # Assign job to GPU
        torch.cuda.set_device(gpu_index)
    
        # Fire up the model for this run
        model, tokenizer, gen_cfg = start_llm(
            gpu=gpu, 
            quantization=quantization
        )

        rows = helper_funcs.get_rows(
            db_name=db_name, 
            user=user, 
            passwd=passwd, 
            host=host, 
            num_abstracts=num_abstracts
        )

        batch_count = 0

        for i in range(batches):

            batch_count += 1
            print(f' Summarizing batch {batch_count} of {num_abstracts // batch_size}.')

            # Get the batch
            batch = rows[i*batch_size:(i+1)*batch_size]

            # Get abstract texts for this batch
            abstracts = [row[1] for row in batch]

            # Do the summaries
            summaries = helper_funcs.summarize(
                abstracts=abstracts,
                model=model,
                tokenizer=tokenizer, 
                gen_cfg=gen_cfg,
                use_GPU=True
            )

            print(f' Job {i}: finished batch {batch_count} of {batches}: {len(batch)} abstracts.')

    except torch.cuda.OutOfMemoryError as oom:

        print(f'{oom}')
        return False

    except RuntimeError as rte:

        print(f'{rte}')
        return False

    # Get rid of model and tokenizer from run, free up memory
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    print(f' Job {1}: done.')

    return True

def start_llm(
    gpu: str, 
    quantization: str
) -> Tuple[
    transformers.T5ForConditionalGeneration, 
    transformers.T5TokenizerFast, 
    transformers.GenerationConfig
]:

    # Set quantization configuration
    quantization_config = transformers.BitsAndBytesConfig(
        load_in_4bit=False
    )

    if quantization == 'four bit':
        quantization_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_compute_dtype=torch.float16
        )

    # Initialize the tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained('haining/scientific_abstract_simplification')

    # Initialize model with selected device map
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
        'haining/scientific_abstract_simplification', 
        device_map = gpu,
        quantization_config=quantization_config
    )
    
    # Load generation config from model and set some parameters as desired
    gen_cfg = transformers.GenerationConfig.from_model_config(model.config)
    gen_cfg.max_length = 256
    gen_cfg.top_p = 0.9
    gen_cfg.do_sample = True

    return model, tokenizer, gen_cfg