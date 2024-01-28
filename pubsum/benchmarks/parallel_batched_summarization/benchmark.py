import gc
from typing import List, Tuple
import time
import torch
import itertools
import multiprocessing as mp
import numpy as np
from .. import helper_functions as helper_funcs
import transformers

def benchmark(
    resume: str,
    results_dir: str,
    replicates: int,
    batches: int,
    batch_size_lists: List[List[int]],
    worker_count_lists: List[List[int]],
    gpus: List[str],
    quantization_strategy_lists: List[List[str]],
    db_name: str, 
    user: str, 
    passwd: str, 
    host: str,
    file_name: str = 'results.csv'
) -> bool:
    
    print(f'\nRunning data parallel, batched summarization benchmark. Resume = {resume}.')

    # Set list of keys for the data we want to collect
    collection_vars = [
        'abstracts',
        'batches',
        'replicate',
        'batch size',
        'workers',
        'quantization',
        'summarization time (sec.)',
        'summarization rate (abstracts/sec.)',
        'max memory allocated (bytes)',
        'model memory footprint (bytes)'
    ]

    # Subset of collection vars which are sufficient to uniquely identify each run
    unique_collection_vars = [
        'quantization',
        'batch size',
        'workers',
        'replicate'
    ]

    # Handel resume request by reading or emptying old data, as appropriate
    completed_runs = helper_funcs.resume_run(
        resume=resume, 
        results_dir=results_dir,
        collection_vars=collection_vars,
        unique_collection_vars=unique_collection_vars,
        file_name=file_name
    )

    # Construct parameter sets
    for batch_sizes, worker_counts, quantization_strategies in zip(batch_size_lists, worker_count_lists, quantization_strategy_lists):
        replicate_numbers = list(range(1, replicates + 1))

        parameter_sets = itertools.product(
            quantization_strategies,
            batch_sizes,
            worker_counts,
            replicate_numbers
        )

        # List to hold parameter sets that cause out-of-memory crashes
        # (quantization, workers, batch_size) - replicate number omitted
        oom_parameter_sets = []

        if len(quantization_strategies) * len(worker_counts) * len(batch_sizes) * len(replicate_numbers) == len(completed_runs):
            print('Run is complete')
        
        else:

            # Loop on parameter sets to run jobs
            for parameter_set in parameter_sets:

                # Check if we have already completed this parameter set
                if parameter_set not in completed_runs:

                    # Unpack parameters from set
                    quantization, batch_size, workers, replicate = parameter_set

                    # Calculate total abstracts needed for job
                    num_abstracts = batches * batch_size

                    # Print run parameters
                    print(f'\nParallel batched summarization:\n')
                    print(f' Replicate: {replicate}')
                    print(f' Model quantization: {quantization}')
                    print(f' Batch size: {batch_size}')
                    print(f' Batches: {batches}')
                    print(f' Total workers: {workers}\n')

                    # Instantiate results object for this run
                    results = helper_funcs.Results(
                        results_dir=results_dir,
                        collection_vars=collection_vars,
                        file_name=file_name
                    )

                    # Collect data for run parameters
                    results.data['abstracts'].append(num_abstracts)
                    results.data['replicate'].append(replicate)
                    results.data['batches'].append(batches)
                    results.data['batch size'].append(batch_size)
                    results.data['workers'].append(workers)
                    results.data['quantization'].append(quantization)

                    # Then, check to see if this parameter set has caused an
                    # out of memory crash on a previous replicate, if it has
                    # skip it and add OOM flag to results directly. To do this
                    # we need to omit the replicate number and compare only on
                    # quantization, workers and batch size
                    if parameter_set[:-1] in oom_parameter_sets:

                        print(' Skipping parameter set due to prior OOMs.')

                        results.data['summarization time (sec.)'].append('OOM')
                        results.data['summarization rate (abstracts/sec.)'].append('OOM')
                        results.data['max memory allocated (bytes)'].append('OOM')
                        results.data['model memory footprint (bytes)'].append('OOM')
                    
                    # If we have not crashed on this parameter set before, do the run
                    else:

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

                            # Increment GPU index - when the GPU index gets to one less than the total 
                            # number of GPUs, reset it back to 0, otherwise increment it.
                            if gpu_index == len(gpus) - 1:
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
                        
                        # Get model and max memory footprint across workers
                        run_total_max_memory = 0
                        run_total_model_footprint = 0

                        for worker_result in result:
                            run_total_max_memory += worker_result[1]
                            run_total_model_footprint += worker_result[2]

                        print(f'\n Worker returns: {result}')
                        print(f' Max memory: {run_total_max_memory}')
                        print(f' Total model memory footprint: {run_total_model_footprint}')

                        # Collect and save timing and memory data, if we returned an OOM error, mark it in the results
                        # and save the parameter set that caused the crash
                        if False not in sum(list(result), []):
                            results.data['summarization time (sec.)'].append(dT)
                            results.data['summarization rate (abstracts/sec.)'].append((batches * batch_size * workers)/dT)
                            results.data['max memory allocated (bytes)'].append(run_total_max_memory)
                            results.data['model memory footprint (bytes)'].append(run_total_model_footprint)
                        else:
                            results.data['summarization time (sec.)'].append('OOM')
                            results.data['summarization rate (abstracts/sec.)'].append('OOM')
                            results.data['max memory allocated (bytes)'].append('OOM')
                            results.data['model memory footprint (bytes)'].append('OOM')

                            oom_parameter_sets.append((quantization, workers, batch_size))

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

        for n in range(batches):

            batch_count += 1
            print(f' Job {i}: summarizing batch {batch_count} of {batches}.')

            # Get the batch
            batch = rows[n*batch_size:(n+1)*batch_size]

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

            print(f' Job {i}: finished batch {batch_count} of {batches}')

    except torch.cuda.OutOfMemoryError as oom:

        print(f'{oom}')
        return [False, np.nan, np.nan]

    except RuntimeError as rte:

        print(f'{rte}')
        return [False, np.nan, np.nan]
    
    # Get max and model memory footprints
    max_memory = torch.cuda.max_memory_allocated()
    model_memory = model.get_memory_footprint()

    # Get rid of model and tokenizer from run, free up memory
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    print(f' Job {i}: done.')

    return [True, max_memory, model_memory]

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

    elif quantization == 'four bit nf4':
        quantization_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float16
        )

    # Initialize the tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        'haining/scientific_abstract_simplification'
    )

    # Initialize model with selected device map
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
        'haining/scientific_abstract_simplification', 
        device_map=gpu,
        quantization_config=quantization_config
    )
    
    # Load generation config from model and set some parameters as desired
    gen_cfg = transformers.GenerationConfig.from_model_config(model.config)
    gen_cfg.max_length = 256
    gen_cfg.top_p = 0.9
    gen_cfg.do_sample = True

    return model, tokenizer, gen_cfg