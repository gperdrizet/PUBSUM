import os
import gc
import pandas as pd
import psycopg2
import time
import torch
import itertools
from typing import List, Tuple
import transformers

# Silence parallelism warning
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

def benchmark(
    helper_funcs,
    resume: bool,
    results_dir: str,
    replicates: int,
    num_abstracts: int,
    quantization_strategies: List[str], 
    db_name: str, 
    user: str, 
    passwd: str, 
    host: str
):
    
    print(f'\nRunning model quantization benchmark. Resume = {resume}.')

    # Set list of keys for the data we want to collect
    collection_vars = [
        'replicate',
        'abstracts',
        'quantization strategy',
        'summarization time (sec.)',
        'summarization rate (abstracts/sec.)',
        'model GPU memory footprint (bytes)',
        'max memory allocated (bytes)'
    ]

    # Subset of independent vars which are sufficient to uniquely identify each run
    unique_collection_vars = [
        'quantization strategy',
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
        replicate_numbers
    )

    if len(quantization_strategies) * len(replicate_numbers) == len(completed_runs):
        print('Run is complete')
    
    else:

        # Loop on parameter sets
        for parameter_set in parameter_sets:

            # Check if we have already completed this parameter set
            if parameter_set not in completed_runs:

                # Unpack parameters from set
                quantization_strategy, replicate = parameter_set

                # Calculate total abstracts needed for job
                num_abstracts = replicates

                print(f'\nModel quantization benchmark:\n')
                print(f' Replicate: {replicate} of {replicates}')
                print(f' Model quantization: {quantization_strategy}')

                # Instantiate results object for this run
                results = helper_funcs.Results(
                    results_dir=results_dir,
                    collection_vars=collection_vars
                )

                # Fire up the model for this run
                model, tokenizer, gen_cfg = start_llm(quantization_strategy)
                model_memory_footprint = model.get_memory_footprint()

                # Get rows from abstracts table
                rows = helper_funcs.get_rows(
                    db_name=db_name, 
                    user=user, 
                    passwd=passwd, 
                    host=host, 
                    num_abstracts=num_abstracts
                )

                # Do and time the summary
                summarization_start = time.time()

                # Loop on rows
                row_count = 0

                for row in rows:

                    row_count += 1

                    print(f' Summarizing abstract: {row_count} of {num_abstracts}')

                    # Get abstract text for this row
                    abstract = row[1]

                    summary = helper_funcs.summarize(
                        abstracts=[abstract], 
                        model=model, 
                        tokenizer=tokenizer, 
                        gen_cfg=gen_cfg, 
                        use_GPU=True
                    )

                dT = time.time() - summarization_start

                # Get max memory used and reset
                max_memory = torch.cuda.max_memory_allocated()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                # Collect data
                results.data['replicate'].append(replicate)
                results.data['abstracts'].append(num_abstracts)
                results.data['quantization strategy'].append(quantization_strategy)
                results.data['model GPU memory footprint (bytes)'].append(model_memory_footprint)
                results.data['max memory allocated (bytes)'].append(max_memory)
                results.data['summarization time (sec.)'].append(dT)
                results.data['summarization rate (abstracts/sec.)'].append(num_abstracts/dT)

                # Get rid of model and tokenizer from run, free up memory
                del model
                del tokenizer
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                # Save the result
                results.save_result()
                print(' Done.')

    return True


def start_llm(quantization_strategy: str) -> Tuple[
    transformers.T5ForConditionalGeneration, 
    transformers.T5TokenizerFast, 
    transformers.GenerationConfig
]:
        
        # Place model on single GPU
        device_map = 'cuda:0'
        
        # Set quantization for bnb
        if 'none' in quantization_strategy:

            # Initialize the tokenizer
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                "haining/scientific_abstract_simplification"
            )

            # Initialize model
            model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
                "haining/scientific_abstract_simplification", 
                device_map=device_map
            )

        elif 'none' not in quantization_strategy:
            if 'eight bit' in quantization_strategy:

                quantization_config = transformers.BitsAndBytesConfig(
                    load_in_8bit=True, 
                    bnb_4bit_compute_dtype=torch.float16
                )

            elif 'four bit' in quantization_strategy and 'nf4' not in quantization_strategy and 'nested' not in quantization_strategy:

                quantization_config = transformers.BitsAndBytesConfig(
                    load_in_4bit=True, 
                    bnb_4bit_compute_dtype=torch.float16
                )

            elif 'four bit nf4' in quantization_strategy and 'nested' not in quantization_strategy:

                quantization_config = transformers.BitsAndBytesConfig(
                    load_in_4bit=True, 
                    bnb_4bit_quant_type='nf4',
                    bnb_4bit_compute_dtype=torch.float16
                )

            elif 'nested four bit' in quantization_strategy and 'nf4' not in quantization_strategy:

                quantization_config = transformers.BitsAndBytesConfig(
                    load_in_4bit=True, 
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16
                )

            elif 'nested four bit nf4' in quantization_strategy:

                quantization_config = transformers.BitsAndBytesConfig(
                    load_in_4bit=True, 
                    bnb_4bit_quant_type='nf4',
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16
                )

            # Initialize the tokenizer
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                'haining/scientific_abstract_simplification'
            )

            # Initialize model
            model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
                'haining/scientific_abstract_simplification', 
                device_map=device_map,
                quantization_config=quantization_config
            )

        if 'BT' in quantization_strategy:

            model.to_bettertransformer()
        
        # Load generation config from model and set some parameters as desired
        gen_cfg = transformers.GenerationConfig.from_model_config(model.config)
        gen_cfg.max_length = 256
        gen_cfg.top_p = 0.9
        gen_cfg.do_sample = True

        return model, tokenizer, gen_cfg