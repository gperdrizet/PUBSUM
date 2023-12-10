import os
import gc
import pandas as pd
import psycopg2
import time
import torch
import itertools
from typing import List, Tuple
import transformers

def benchmark(
    helper_funcs,
    resume: bool, 
    results_dir: str, 
    replicates: int,
    num_abstracts: int,
    device_map_strategies: List[str],
    db_name: str,
    user: str,
    passwd: str,
    host: str
):
    
    print(f'\nRunning huggingface device map benchmark. Resume = {resume}.\n')

    # Set list of keys for the data we want to collect
    collection_vars = [
        'replicate',
        'abstracts',
        'device map strategy',
        'summarization time (sec.)',
        'summarization rate (abstracts/sec.)'
    ]

    # Subset of independent vars which are sufficient to uniquely identify each run
    unique_collection_vars = [
        'device map strategy',
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
        device_map_strategies,
        replicate_numbers
    )

    # Loop on parameter sets
    for parameter_set in parameter_sets:

        # Check if we have already completed this parameter set
        if parameter_set not in completed_runs:

            # Unpack parameters from set
            device_map_strategy, replicate = parameter_set

            print(f'\nHF device map strategy benchmark:\n')
            print(f' Replicate: {replicate} of {replicates}')
            print(f' Device map strategy: {device_map_strategy}')

            # Instantiate results object for this run
            results = helper_funcs.Results(
                results_dir=results_dir,
                collection_vars=collection_vars
            )

            # Fire up the model for this run
            model, tokenizer, gen_cfg = start_llm(device_map_strategy=device_map_strategy)

            # Get rows from abstracts table
            rows = helper_funcs.get_rows(
                db_name=db_name, 
                user=user, 
                passwd=passwd, 
                host=host, 
                num_abstracts=num_abstracts
            )

            # Do and time the summaries
            summarization_start = time.time()

            # Loop on rows
            row_count = 0

            for row in rows:

                row_count += 1
                print(f' Summarizing abstract: {row_count} of {num_abstracts}')

                # Get abstract text for this row
                abstract = row[1]

                # Decide if we need to move encoding to GPU
                if device_map_strategy != 'CPU only':
                    use_GPU = True

                else:
                    use_GPU = False

                summary = helper_funcs.summarize(
                    abstracts=[abstract], 
                    model=model, 
                    tokenizer=tokenizer, 
                    gen_cfg=gen_cfg, 
                    use_GPU=use_GPU
                )

            dT = time.time() - summarization_start

            # Collect results
            results.data['replicate'].append(replicate)
            results.data['abstracts'].append(num_abstracts)
            results.data['device map strategy'].append(device_map_strategy)
            results.data['summarization time (sec.)'].append(dT)
            results.data['summarization rate (abstracts/sec.)'].append(1/dT)

            # Save the result
            results.save_result()

            # Get rid of model and tokenizer from run, free up memory
            del model
            del tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            
            print(' Done.')


def start_llm(device_map_strategy: str) -> Tuple[
    transformers.T5ForConditionalGeneration, 
    transformers.T5TokenizerFast, 
    transformers.GenerationConfig
]:
        
        # Translate device map strategy for huggingface. Start with
        # device map set to CPU only by default
        device_map = 'cpu'

        if device_map_strategy == 'multi-GPU':
            device_map = 'auto'

        elif device_map_strategy == 'single GPU':
            device_map = 'cuda:0'

        elif device_map_strategy == 'balanced':
            device_map = 'balanced'

        elif device_map_strategy == 'balanced_low_0':
            device_map = 'balanced_low_0'

        elif device_map_strategy == 'sequential':
            device_map = 'sequential'

        model = transformers.AutoModelForSeq2SeqLM.from_pretrained("haining/scientific_abstract_simplification", device_map = device_map)
        
        # Load generation config from model and set some parameters as desired
        gen_cfg = transformers.GenerationConfig.from_model_config(model.config)
        gen_cfg.max_length = 256
        gen_cfg.top_p = 0.9
        gen_cfg.do_sample = True

        # Initialize the tokenizer
        tokenizer = transformers.AutoTokenizer.from_pretrained("haining/scientific_abstract_simplification")

        return model, tokenizer, gen_cfg