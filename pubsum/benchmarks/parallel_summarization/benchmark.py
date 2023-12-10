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
    
    print(f'\nRunning data parallel summarization benchmark. Resume = {resume}.\n')

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

            # Give torch CPU threads based on device and workers for this run, if appropriate
            if device == 'CPU':
                torch.set_num_threads(1)

            # If this is a GPU run, start a counter to pick GPUs for jobs and give the
            # GPU access to all CPU physical cores
            if device == 'GPU':
                gpu_index = 0
                torch.set_num_threads(mp.cpu_count())

            # If this run is not using GPU(s), pass none for GPU related parameters
            else:
                gpu_index = None
                gpu = None

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
                            gpu
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
    gpu: str
) -> bool:

    try:
        # Assign job to GPU if needed
        if device == 'GPU':
            use_GPU = True
            torch.cuda.set_device(gpu_index)
            print(f' Job {i} starting on {gpu}')

        else:
            use_GPU = False
            print(f' Job {i} starting on CPU')
        
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

# def summarize(abstract, model, tokenizer, gen_cfg, device_map_strategy):
        
#     # Prepend the prompt to this abstract and encode
#     encoding = tokenizer(
#         'summarize, simplify, and contextualize: ' + abstract, 
#         max_length = 672, 
#         padding = 'max_length', 
#         truncation = True, 
#         return_tensors = 'pt'
#     )

#     # Move to GPU if appropriate
#     if 'CPU' not in device_map_strategy.split(' '):
#         encoding = encoding.to('cuda')
    
#     # Generate summary
#     decoded_ids = model.generate(
#         input_ids = encoding['input_ids'],
#         attention_mask = encoding['attention_mask'], 
#         generation_config = gen_cfg
#     )
    
#     # Decode summary
#     summary = tokenizer.decode(decoded_ids[0], skip_special_tokens = True)

#     return summary

# def collect_result(result):
#     # Need a dummy return here since we don't have good logging setup
#     # this ensures that anything that a worker process sends to STDOUT
#     # or STDERR actually shows up in the terminal
#     return True

# class Results:
#     '''Class to hold objects and methods for
#     collection of results'''

#     def __init__(self, results_dir):

#         # Output file for results
#         self.output_file = f'{results_dir}/results.csv'

#         # Independent vars for run
#         self.data = {}
#         self.data['replicate'] = []
#         self.data['abstracts'] = []
#         self.data['abstracts per worker'] = []
#         self.data['workers'] = []
#         self.data['device map strategy'] = []
#         self.data['summarization time (sec.)'] = []
#         self.data['summarization rate (abstracts/sec.)'] = []

#     def save_result(self, overwrite = False):

#         # Make dataframe of new results
#         results_df = pd.DataFrame(self.data)

#         # Read existing results if any and concatenate new results if desired
#         if overwrite == False:
#             if os.path.exists(self.output_file):
#                 old_results_df = pd.read_csv(self.output_file)
#                 results_df = pd.concat([old_results_df, results_df])

#         else:
#             print('Clearing any old results.')

#         # Save results for run to csv
#         results_df.to_csv(self.output_file, index = False)