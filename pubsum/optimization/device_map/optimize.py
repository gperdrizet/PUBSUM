import os
import gc
import pandas as pd
import psycopg2
import time
import torch
import itertools
from typing import List, Tuple
import transformers

def optimize(
    helper_funcs,
    resume: bool, 
    output_dir: str,
    output_filename: str,
    replicates: int,
    device_map_strategies: List[str],
    db_name: str,
    user: str,
    passwd: str,
    host: str
):
    
    print(f'\nRunning device map optimization with resume = {resume}.')

    # Instantiate results object and resume data collection if asked
    results=helper_funcs.Results(
        independent_vars=['device map strategy', 'replicate'],
        dependent_vars=['summarization time (sec.)', 'summarization rate (abstracts/sec.)']
        output_dir=output_dir, 
        output_filename=output_filename,
        resume=resume
    )

    # Construct parameter sets for run
    replicate_numbers = list(range(1, replicates + 1))
    parameter_sets = itertools.product(device_map_strategies, replicate_numbers)


    # If the resumed results already contain all of the runs we are going to do
    # everyone can go home
    if len(device_map_strategies) * len(replicate_numbers) == len(results.completed_runs):
        print('Benchmark is complete')
    
    else:

        # Track previous device map, so that we only bother reloading the model when we need
        # to switch device maps, since we are entering the first run, set to None
        previous_run_device_map = None
        model = None
        tokenizer = None

        # Loop on parameter sets
        for parameter_set in parameter_sets:

            # Check if we have already completed this parameter set
            if parameter_set not in results.completed_runs:

                # Unpack parameters from set
                device_map_strategy, replicate = parameter_set

                print(f'\nDevice map strategy optimization:\n')
                print(f' Replicate: {replicate} of {replicates}')
                print(f' Device map strategy: {device_map_strategy}')

                # Check if we are switching device maps from the previous run
                if device_map_strategy != previous_run_device_map:
                    print(' Initializing model')

                    # Get rid of old model and tokenizer from run, free up memory
                    del model
                    del tokenizer
                    gc.collect()
                    torch.cuda.empty_cache()

                    # Decide if we need to move encoding to GPU
                    if device_map_strategy != 'CPU only':
                        use_GPU = True

                    else:
                        use_GPU = False

                    # Fire up the model for this run
                    model, tokenizer, gen_cfg = start_llm(device_map_strategy=device_map_strategy)

                    # Update the previous device map for next iteration
                    previous_run_device_map = device_map_strategy
                
                else:
                    print(' Reusing model')

                # Get row from abstracts table
                row = helper_funcs.get_rows(
                    db_name=db_name, 
                    user=user, 
                    passwd=passwd, 
                    host=host, 
                    num_abstracts=1
                )

                # Get abstract text for this row
                abstract = row[0][1]

                # Do and time the summary
                summarization_start = time.time()

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
                results.data['device map strategy'].append(device_map_strategy)
                results.data['summarization time (sec.)'].append(dT)
                results.data['summarization rate (abstracts/sec.)'].append(1/dT)

                # Save the result and reinitialize collector every 5 runs
                if replicate % 5 == 0:
                    results.save_result()
                    results.clear()
                    
                print(' Done')

        # Get rid of model and tokenizer from run, free up memory
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    return True


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

        # elif device_map_strategy == 'balanced_low_0':
        #     device_map = 'balanced_low_0'

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