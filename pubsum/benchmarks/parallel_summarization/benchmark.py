import gc
import time
import torch
import itertools
import tracemalloc
import multiprocessing as mp
import numpy as np
from typing import Tuple, List
from .. import helper_functions as helper_funcs
import transformers

def benchmark(
    resume: bool,
    results_dir: str,
    replicates: int,
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
        'replicate',
        'device',
        'workers',
        'summarization time (sec.)',
        'summarization rate (abstracts/sec.)',
        'max memory allocated (bytes)',
        'model memory footprint (bytes)'
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

    # List to hold parameter sets that cause out-of-memory crashes
    # (device, workers) - replicate number omitted
    oom_parameter_sets = []

    if len(devices) * len(workers) * len(replicate_numbers) == len(completed_runs):
        print('Run is complete')
    
    else:

        # Loop on parameter sets to run jobs
        for parameter_set in parameter_sets:

            # Check if we have already completed this parameter set
            if parameter_set not in completed_runs:

                # Unpack parameters from set
                device, workers, replicate = parameter_set

                # If this is a CPU run, extract the number of threads per worker
                if 'GPU' not in device:
                    threads_per_CPU_worker = int(device.split(' ')[1])

                # If its a GPU run, set threads per CPU worker to none
                elif 'GPU' in device:
                    threads_per_CPU_worker = None

                # Print run parameters
                print(f'\nParallel summarization:\n')
                print(f' Replicate: {replicate}')
                print(f' Device: {device}')
                print(f' Workers: {workers}\n')

                # If this is a CPU run and it's calling for more threads than we have, skip it
                if 'GPU' not in device and mp.cpu_count() < workers * threads_per_CPU_worker:
                    print(f' Run will use more worker threads than total threads, skipping')

                else:

                    # Instantiate results object for this run
                    results = helper_funcs.Results(
                        results_dir=results_dir,
                        collection_vars=collection_vars
                    )

                    # Collect data for run parameters
                    results.data['replicate'].append(replicate)
                    results.data['workers'].append(workers)
                    results.data['device'].append(device)

                    # Then, check to see if this parameter set has caused an
                    # out of memory crash on a previous replicate, if it has
                    # skip it and add OOM flag to results directly. To do this
                    # we need to omit the replicate number and compare only on
                    # quantization, workers and batch size
                    if parameter_set[:-1] in oom_parameter_sets:

                        print(' Skipping parameter set due to prior OOMs.')

                        results.data['summarization time (sec.)'].append('OOM')
                        results.data['summarization rate (abstracts/sec.)'].append('OOM')
                        results.data['model memory footprint (bytes)'].append('OOM')
                        results.data['max memory allocated (bytes)'].append('OOM')

                    # If we have not crashed on this parameter set before, do the run
                    else:

                        # If this is a one GPU per job run, start a counter to pick GPUs for jobs
                        if device == 'GPU':
                            gpu_index = 0

                        # If we are using the GPUs in sequential mode, pass that for 'gpu' and no
                        # GPU index
                        if device == 'GPU: sequential':
                            gpu_index = None
                            gpu = 'sequential'

                        # If this run is not using GPU(s), pass none for GPU related parameters,
                        # and get the number of CPU threads per worker called for.
                        else:
                            gpu_index = None
                            gpu = None

                        # Don't do the run if total CPU threads required is more than available
                        if 'GPU' in device or mp.cpu_count() // workers >= threads_per_CPU_worker:

                            # If this is a CPU run, start tracemalloc to monitor
                            # system memory use. Getting this from torch does
                            # not give a sane value.
                            # if 'CPU' in device:
                            #     tracemalloc.start()

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
                                            device, 
                                            gpu_index, 
                                            gpu,
                                            threads_per_CPU_worker
                                        )
                                    )
                                )

                                # Increment GPU index if needed - if if gpu_index is one less than the number 
                                # of gpus, reset it back to 0, otherwise increment it.
                                if device == 'GPU':
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

                            # Collect max memory and model footprint from workers
                            total_max_memory = 0
                            total_model_footprint = 0

                            for worker_result in result:
                                total_max_memory += worker_result[1]
                                total_model_footprint += worker_result[2]

                            print(f'\n Worker returns: {result}')
                            print(f' Total max memory: {round(total_max_memory / (1024**3), 2)}')
                            print(f' Total model memory footprint: {round(total_model_footprint / (1024**3), 2)}')

                            # Collect and save timing and memory data, if we returned an OOM error, mark it in the results
                            # and save the parameter set that caused the crash
                            if False not in sum(list(result), []):
                                
                                results.data['summarization time (sec.)'].append(dT)
                                results.data['summarization rate (abstracts/sec.)'].append((workers)/dT)
                                results.data['model memory footprint (bytes)'].append(total_model_footprint)
                                results.data['max memory allocated (bytes)'].append(total_max_memory)

                            else:
                                results.data['summarization time (sec.)'].append('OOM')
                                results.data['summarization rate (abstracts/sec.)'].append('OOM')
                                results.data['model memory footprint (bytes)'].append('OOM')
                                results.data['max memory allocated (bytes)'].append('OOM')

                                oom_parameter_sets.append((device, workers))
                            
                    results.save_result()
    return True


def start_job(
    i: int, 
    db_name: str, 
    user: str, 
    passwd: str, 
    host: str,  
    device: str, 
    gpu_index: int, 
    gpu: str,
    threads_per_CPU_worker: int
) -> List:

    try:
        # Assign job to specific GPU if needed
        if device == 'GPU':
            use_GPU = True
            torch.cuda.set_device(gpu_index)
            print(f' Job {i} starting on {gpu}')

        # If running on GPUs in sequential mode, don't set a specific GPU
        elif device == 'GPU: sequential':
            use_GPU = True
            print(f' Job {i} starting on GPUs in {gpu} mode')        

        # If this is not a GPU run, assign CPU threads
        # and start tracemalloc to monitor system memory
        else:
            use_GPU = False
            torch.set_num_threads(threads_per_CPU_worker)
            tracemalloc.start()
            print(f' Job {i} starting on CPU with {threads_per_CPU_worker} threads per worker')
        
        # Fire up the model for this run
        model, tokenizer, gen_cfg = start_llm(gpu=gpu)

        # Get an abstract to summarize
        row = helper_funcs.get_rows(
            db_name=db_name, 
            user=user, 
            passwd=passwd, 
            host=host, 
            num_abstracts=1
        )

        # Get abstract text
        abstract = row[0][1]

        # Do the summary
        summary = helper_funcs.summarize(
            abstracts=[abstract],
            model=model,
            tokenizer=tokenizer, 
            gen_cfg=gen_cfg,
            use_GPU=use_GPU
        )

    except torch.cuda.OutOfMemoryError as oom:

        print(f'{oom}')
        return [False, np.nan, np.nan]

    except RuntimeError as rte:

        print(f'{rte}')
        return [False, np.nan, np.nan]
    
    # Get peak memory for GPU, return None for CPU because
    # we are tracking system memory around the apply_async
    # call rather than using torch.cuda 
    if 'GPU' in device:
        max_memory = torch.cuda.max_memory_allocated()

    else:
        memory, max_memory = tracemalloc.get_traced_memory()
        max_memory = max_memory * 1024
        tracemalloc.stop()

    model_memory = model.get_memory_footprint()

    # Get rid of model and tokenizer from run, free up memory
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    print(f' Job {i}: done.')

    return [True, max_memory, model_memory]


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