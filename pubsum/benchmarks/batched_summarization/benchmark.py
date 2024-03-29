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
    batch_size_lists: List[List[int]], 
    quantization_strategy_lists: List[List[str]],
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
        'model memory footprint (bytes)',
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
    for batch_sizes, quantization_strategies in zip(batch_size_lists, quantization_strategy_lists):
        replicate_numbers = list(range(1, replicates + 1))

        parameter_sets = itertools.product(
            quantization_strategies,
            batch_sizes,
            replicate_numbers
        )

        # List to hold parameter sets that cause out-of-memory crashes
        # (device, workers) - replicate number omitted
        oom_parameter_sets = []

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
                                print(f' Summarizing batch {batch_count} of {batches}.')

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

                            # Get max memory
                            max_memory = torch.cuda.max_memory_allocated()

                            # Collect run results
                            results.data['model memory footprint (bytes)'].append(model_memory_footprint)
                            results.data['max memory allocated (bytes)'].append(max_memory)
                            results.data['summarization time (sec.)'].append(dT)
                            results.data['summarization rate (abstracts/sec.)'].append(num_abstracts/dT)

                        # Catch out of memory errors
                        except torch.cuda.OutOfMemoryError as oom:
                            print(f'\n {oom}\n')

                            # Since we failed on OOM, mark it in results
                            results.data['model memory footprint (bytes)'].append('OOM')
                            results.data['max memory allocated (bytes)'].append('OOM')
                            results.data['summarization time (sec.)'].append('OOM')
                            results.data['summarization rate (abstracts/sec.)'].append('OOM')

                            # Then save the parameters that caused the OOM
                            oom_parameter_sets.append((quantization, batch_size))

                        # Get rid of model and tokenizer from run
                        del model
                        del tokenizer

                        # Free up memory
                        gc.collect()
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()

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

        elif quantization == 'four bit nf4':
            quantization_config = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
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