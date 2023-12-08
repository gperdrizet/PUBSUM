import os
import pandas as pd
import psycopg2
import time
import torch
import itertools
import multiprocessing as mp
from .. import helper_functions as helper_funcs
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig, BitsAndBytesConfig

def benchmark(
    resume: str,
    results_dir: str,
    replicates: int,
    batches: int,
    batch_sizes: [int],
    GPU_jobs: [int],
    gpus: [str],
    quantization_strategies: [str],
    db_name: str, 
    user: str, 
    passwd: str, 
    host: str
) -> bool:
    
    print(f'\nRunning data parallel, batched summarization benchmark. Resume = {resume}.\n')

    # Set list of keys for the data we want to collect
    independent_vars = [
        'abstracts',
        'batches',
        'replicate',
        'batch size',
        'workers',
        'jobs per GPU',
        'quantization',
        'summarization time (sec.)',
        'summarization rate (abstracts/sec.)'
    ]

    # Subset of independent vars which are sufficient to uniquely identify each run
    unique_independent_vars = [
        'quantization',
        'workers',
        'batch size',
        'replicate'
    ]

    # Handel resume request by reading or emptying old data, as appropriate
    completed_runs = helper_funcs.resume_run(
        resume=resume, 
        results_dir=results_dir,
        independent_vars=independent_vars,
        unique_independent_vars=unique_independent_vars
    )

    # Construct parameter sets
    replicate_numbers = list(range(1, replicates + 1))

    parameter_sets = itertools.product(
        quantization_strategies,
        GPU_jobs,
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
                independent_vars=independent_vars
            )

            # Collect data for run parameters
            results.data['abstracts'].append(num_abstracts)
            results.data['replicate'].append(replicate)
            results.data['batches'].append(batches)
            results.data['batch size'].append(batch_size)
            results.data['workers'].append(workers)
            results.data['jobs per GPU'].append(workers // 4)
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

            async_results = []

            # Loop on jobs for this run
            for i in list(range(0, workers)):

                # Pick GPU
                gpu = gpus[gpu_index]

                async_results.append(
                    pool.apply_async(start_job,
                        args = (i, db_name, user, passwd, host, batches, batch_size, gpu_index, gpu, quantization)
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
            print(f' Async results: {result}')

            # Collect and save data, if we returned an OOM error, mark it in the results
            if False not in result:
                results.data['summarization time (sec.)'].append(dT)
                results.data['summarization rate (abstracts/sec.)'].append((batches * batch_size * workers)/dT)

            else:
                results.data['summarization time (sec.)'].append('OOM')
                results.data['summarization rate (abstracts/sec.)'].append('OOM')

            results.save_result()


def start_job(i, db_name, user, passwd, host, batches, batch_size, gpu_index, gpu, quantization):
    
    print(f' Job {i}: starting on {gpu}.')

    try:
        # Assign job to GPU
        torch.cuda.set_device(gpu_index)
    
        # Fire up the model for this run
        model, tokenizer, gen_cfg = start_llm(gpu, quantization)

        # Get abstracts to summarize
        num_abstracts = batch_size * batches
        rows = helper_funcs.get_rows(db_name, user, passwd, host, num_abstracts)

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

    print(f' Job {1}: done.')

    return True

# def get_rows(db_name, user, passwd, host, num_abstracts):
        
#     # Open connection to PUBMED database on postgreSQL server, create connection
#     connection = psycopg2.connect(f'dbname={db_name} user={user} password={passwd} host={host}')

#     # Start new reader cursor
#     read_cursor = connection.cursor()

#     # Loop until we have num_abstracts non-empty rows to return. Note: ideally we would go back to
#     # the article parsing script and not put empty abstracts into the SQL database. Let's do
#     # that later, but this will work for now to get us were we want to go. Also, this is not
#     # being timed as part of the benchmark, so any inefficacy in selecting a few hundred abstracts
#     # is irrelevant

#     # Get 2x the number of rows we want
#     read_cursor.execute('SELECT * FROM abstracts ORDER BY random() LIMIT %s', (num_abstracts*2,))

#     # Collect non-empty rows until we have enough
#     rows = []

#     for row in read_cursor:

#         abstract = row[1]

#         if abstract != None:
#             rows.append(row)

#         if len(rows) == num_abstracts:
#             break

#     read_cursor.close()

#     return rows


def start_llm(gpu, quantization):

    # Set quantization configuration
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=False
    )

    if quantization == 'four bit':
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=False, 
            bnb_4bit_compute_dtype=torch.float16
        )

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained('haining/scientific_abstract_simplification')

    # Initialize model with selected device map
    model = AutoModelForSeq2SeqLM.from_pretrained(
        'haining/scientific_abstract_simplification', 
        device_map = gpu,
        quantization_config=quantization_config
    )
    
    # Load generation config from model and set some parameters as desired
    gen_cfg = GenerationConfig.from_model_config(model.config)
    gen_cfg.max_length = 256
    gen_cfg.top_p = 0.9
    gen_cfg.do_sample = True

    return model, tokenizer, gen_cfg

# def summarize(abstracts, model, tokenizer, gen_cfg):

#     # Prepend the prompt to this abstract and encode
#     inputs = ['summarize, simplify, and contextualize: ' + abstract for abstract in abstracts]
        
#     encoding = tokenizer(
#         inputs, 
#         max_length = 672, 
#         padding = 'max_length', 
#         truncation = True, 
#         return_tensors = 'pt'
#     )

#     encoding = encoding.to('cuda')
    
#     # Generate summary
#     decoded_ids = model.generate(
#         input_ids = encoding['input_ids'],
#         attention_mask = encoding['attention_mask'], 
#         generation_config = gen_cfg
#     )
    
#     # Decode summary
#     summaries = tokenizer.decode(decoded_ids[0], skip_special_tokens = True)

#     return summaries

# class Results:
#     '''Class to hold objects and methods for
#     collection of results'''

#     def __init__(self, results_dir):

#         # Output file for results
#         self.output_file = f'{results_dir}/results.csv'

#         # Independent vars for run
#         self.data = {}
#         self.data['abstracts'] = []
#         self.data['batches'] = []
#         self.data['replicate'] = []
#         self.data['batch size'] = []
#         self.data['workers'] = []
#         self.data['jobs per GPU'] = []
#         self.data['quantization'] = []
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

# def generate_batches(lst, n):
#     '''Yield successive n-sized chunks from lst.'''
#     for i in range(0, len(lst), n):
#         yield lst[i:i + n]