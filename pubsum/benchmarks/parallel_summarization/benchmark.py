import os
import gc
import pandas as pd
import psycopg2
import time
import torch
import itertools
import multiprocessing as mp
from typing import Tuple, List
from .. import helper_functions as helper_funcs
import transformers

def benchmark(
    resume: bool,
    results_dir: str,
    replicates: int,
    batches: int,
    devices: List[str],
    workers: List[int],
    gpus: List[str],
    db_name: List[str], 
    user: List[str], 
    passwd: List[str], 
    host: List[str]
) -> bool:
    
    print(f'\nRunning data parallel summarization benchmark. Resume = {resume}.')

    # Set list of keys for the data we want to collect
    collection_vars = [
        'abstracts',
        'abstracts per worker',
        'replicate',
        'batches',
        'device',
        'workers',
        'summarization time (sec.)',
        'summarization rate (abstracts/sec.)'
    ]

    # Subset of collection vars which are sufficient to uniquely identify each run
    unique_collection_vars = [
        'device',
        'workers',
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
        devices,
        workers,
        replicate_numbers
    )

    if len(devices) * len(workers) * len(replicate_numbers) == len(completed_runs):
        print('Run is complete')
    
    else:

        # Loop on parameter sets to run jobs
        for parameter_set in parameter_sets:

            # Check if we have already completed this parameter set
            if parameter_set not in completed_runs:

                # Unpack parameters from set
                device, workers, replicate = parameter_set

                # Calculate total abstracts needed for job
                num_abstracts = batches * workers

                print(f'\nParallel summarization:\n')
                print(f' Replicate: {replicate}')
                print(f' Device: {device}')
                print(f' Workers: {workers}')
                print(f' Batches: {batches}')
                print(f' Abstracts: {num_abstracts}\n')

                # Instantiate results object for this run
                results = helper_funcs.Results(
                    results_dir=results_dir,
                    collection_vars=collection_vars
                )

                # Collect data for run parameters
                results.data['replicate'].append(replicate)
                results.data['abstracts per worker'].append(num_abstracts // workers)
                results.data['abstracts'].append(num_abstracts)
                results.data['batches'].append(batches)
                results.data['workers'].append(workers)
                results.data['device'].append(device)

                # If this is a GPU run, start a counter to pick GPUs for jobs
                if device == 'GPU':
                    gpu_index = 0

                # If this run is not using GPU(s), pass none for GPU related parameters,
                # and get the number of CPU threads per worker called for
                else:
                    gpu_index = None
                    gpu = None
                    threads_per_CPU_worker = int(device.split(' ')[1])

                # Don't do the run if total CPU threads required is more than available
                if device == 'GPU' or mp.cpu_count() // workers >= threads_per_CPU_worker:

                    # Instantiate pool with one member for each job we need to run
                    pool = mp.Pool(
                        processes = workers,
                        maxtasksperchild = 1
                    )

                    # Start timer
                    start = time.time()

                    # Holder for returns from workers
                    async_results = []

                    # Loop on jobs for this run
                    for i in list(range(0, workers)):

                        # Pick GPU for run, if needed
                        if device == 'GPU':
                            gpu = gpus[gpu_index]

                        async_results.append(
                            pool.apply_async(start_job,
                                args = (
                                    i, 
                                    db_name, 
                                    user, 
                                    passwd, 
                                    host, 
                                    batches, 
                                    num_abstracts, 
                                    device, 
                                    gpu_index, 
                                    gpu,
                                    threads_per_CPU_worker
                                )
                            )
                        )

                        # Increment GPU index if needed - we have four GPUs, so when the index gets 
                        # to 3 (0 anchored), reset it back to 0, otherwise increment it.
                        if device == 'GPU':
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
                        results.data['summarization rate (abstracts/sec.)'].append((num_abstracts)/dT)

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
    batches: int, 
    num_abstracts: int, 
    device: str, 
    gpu_index: int, 
    gpu: str,
    threads_per_CPU_worker: int
) -> bool:

    try:
        # Assign job to GPU if needed
        if device == 'GPU':
            use_GPU = True
            torch.cuda.set_device(gpu_index)
            print(f' Job {i} starting on {gpu}')

        else:
            use_GPU = False
            torch.set_num_threads(threads_per_CPU_worker)
            print(f' Job {i} starting on CPU with {threads_per_CPU_worker} threads per worker')
        
        # Fire up the model for this run
        model, tokenizer, gen_cfg = start_llm(gpu=gpu)

        # Get an abstract to summarize
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

            if use_GPU == True:
                print(f' {gpu} summarizing batch {batch_count}.')
                
            else:
                print(f' CPU summarizing batch {batch_count}')

            # Get the batch
            batch = rows[i:(i+1)]

            # Get abstract texts for this batch
            abstracts = [row[1] for row in batch]

            # Do the summaries
            summaries = helper_funcs.summarize(
                abstracts=abstracts,
                model=model,
                tokenizer=tokenizer, 
                gen_cfg=gen_cfg,
                use_GPU=use_GPU
            )

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


def start_llm(gpu: str) -> Tuple[
    transformers.T5ForConditionalGeneration, 
    transformers.T5TokenizerFast, 
    transformers.GenerationConfig
]:
        
        # Set device_map parameter for Huggingface
        if gpu == None:
            device_map = 'cpu'

        else:
            device_map = gpu

        # Initialize model with selected device map
        model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            "haining/scientific_abstract_simplification", 
            device_map = device_map
        )
        
        # Load generation config from model and set some parameters as desired
        gen_cfg = transformers.GenerationConfig.from_model_config(model.config)
        gen_cfg.max_length = 256
        gen_cfg.top_p = 0.9
        gen_cfg.do_sample = True

        # Initialize the tokenizer
        tokenizer = transformers.AutoTokenizer.from_pretrained("haining/scientific_abstract_simplification")

        return model, tokenizer, gen_cfg