import gc
from typing import List, Tuple
import pandas as pd
import time
import torch
import itertools
import transformers

def benchmark(
    helper_funcs,
    resume: str, 
    results_dir: str, 
    replicates: int, 
    batches: int, 
    batch_sizes: List[int], 
    quantization_strategies: List[str],
    db_name: str, 
    user: str, 
    passwd: str, 
    host: str
) -> bool:
    
    print(f'\nRunning batched summarization benchmark. Resume = {resume}.')

    # Set list of keys for the data we want to collect
    collection_vars = [
        'abstracts',
        'replicate',
        'batches',
        'batch size',
        'quantization',
        'summarization time (sec.)',
        'summarization rate (abstracts/sec.)',
        'model GPU memory footprint (bytes)',
        'max memory allocated (bytes)'
    ]

    # Subset of independent vars which are sufficient to uniquely identify each run
    unique_collection_vars = [
        'quantization',
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
        batch_sizes,
        replicate_numbers
    )

    if len(quantization_strategies) * len(batch_sizes) * len(replicate_numbers) == len(completed_runs):
        print('Run is complete')
    
    else:

        # Loop on parameter sets
        for parameter_set in parameter_sets:

            # Check if we have already completed this parameter set
            if parameter_set not in completed_runs:

                # Unpack parameters from set
                quantization, batch_size, replicate = parameter_set

                # Calculate total abstracts needed for job
                num_abstracts = batches * batch_size

                print(f'\nBatched summarization benchmark:\n')
                print(f' Replicate: {replicate} of {replicates}')
                print(f' Model quantization: {quantization}')
                print(f' Batch size: {batch_size}')
                print(f' Batches: {batches}')
                print(f' Abstracts: {num_abstracts}\n')

                # Instantiate results object for this run
                results = helper_funcs.Results(
                    results_dir=results_dir,
                    collection_vars=collection_vars
                )

                # Collect data for run parameters
                results.data['abstracts'].append(num_abstracts)
                results.data['replicate'].append(replicate)
                results.data['quantization'].append(quantization)
                results.data['batches'].append(batches)
                results.data['batch size'].append(batch_size)

                # Fence to catch out of memory errors
                try:
                    # Fire up the model for this run
                    model, tokenizer, gen_cfg = start_llm(quantization=quantization)
                    model_memory_footprint = model.get_memory_footprint()

                    # Get enough rows to batch from abstracts table
                    rows = helper_funcs.get_rows(
                        db_name=db_name, 
                        user=user, 
                        passwd=passwd, 
                        host=host, 
                        num_abstracts=num_abstracts
                    )

                    # Start the batch loop
                    batch_count = 0
                    summarization_start = time.time()

                    for i in range(batches):

                        batch_count += 1
                        print(f' Summarizing batch {batch_count} of {num_abstracts // batch_size}.')

                        # Get the batch
                        batch = rows[i*batch_size:(i+1)*batch_size]

                        # Get abstract texts for this batch
                        abstracts = [row[1] for row in batch]

                        # Do the summary
                        summaries = helper_funcs.summarize(
                            abstracts=abstracts,
                            model=model,
                            tokenizer=tokenizer, 
                            gen_cfg=gen_cfg,
                            use_GPU=True
                        )

                    # Stop the clock
                    dT = time.time() - summarization_start

                    # Get max memory used and reset
                    max_memory = torch.cuda.max_memory_allocated()
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()

                    # Collect run results
                    results.data['model GPU memory footprint (bytes)'].append(model_memory_footprint)
                    results.data['max memory allocated (bytes)'].append(max_memory)
                    results.data['summarization time (sec.)'].append(dT)
                    results.data['summarization rate (abstracts/sec.)'].append(num_abstracts/dT)

                # Catch out of memory errors
                except torch.cuda.OutOfMemoryError as oom:
                    print(f'\n {oom}\n')

                    # Since we failed on OOM, mark it in results
                    results.data['model GPU memory footprint (bytes)'].append('OOM')
                    results.data['max memory allocated (bytes)'].append('OOM')
                    results.data['summarization time (sec.)'].append('OOM')
                    results.data['summarization rate (abstracts/sec.)'].append('OOM')

                # Get rid of model and tokenizer from run, free up memory
                del model
                del tokenizer
                gc.collect()
                torch.cuda.empty_cache()

                # Save the result
                results.save_result()
                print(' Done.')

    return True

def start_llm(quantization: str) -> Tuple[
    transformers.T5ForConditionalGeneration, 
    transformers.T5TokenizerFast, 
    transformers.GenerationConfig
]:
        
        # Place model on single GPU
        device_map = 'cuda:0'

        # Set quantization config
        if quantization == 'none':
            quantization_config = transformers.BitsAndBytesConfig(None)

        elif quantization == 'four bit':
            quantization_config = transformers.BitsAndBytesConfig(
                load_in_4bit=True, 
                bnb_4bit_compute_dtype=torch.float16
            )

        # Initialize the tokenizer
        tokenizer = transformers.AutoTokenizer.from_pretrained('haining/scientific_abstract_simplification')

        # Initialize model
        model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            'haining/scientific_abstract_simplification', 
            device_map=device_map,
            quantization_config=quantization_config
        )

        #model.to_bettertransformer()
        
        # Load generation config from model and set some parameters as desired
        gen_cfg = transformers.GenerationConfig.from_model_config(model.config)
        gen_cfg.max_length = 256
        gen_cfg.top_p = 0.9
        gen_cfg.do_sample = True

        return model, tokenizer, gen_cfg