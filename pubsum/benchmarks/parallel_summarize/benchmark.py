import os
import pandas as pd
import psycopg2
import time
import torch
import multiprocessing as mp
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig

def benchmark(db_name, user, passwd, host, resume, results_dir, num_abstracts, 
              device_map_strategies, num_CPU_jobs, num_GPU_jobs, gpus):
    
    print(f'\nRunning data parallel summarization benchmark. Resume = {resume}\n')

    # If we are resuming a prior run, read old data and collect the
    # completed conditions as a list of lists so we can skip them.
    if resume == 'True':

        if os.path.exists(f'{results_dir}/results.csv'):

            old_results_df = pd.read_csv(f'{results_dir}/results.csv')

            completed_runs = list(zip(
                old_results_df['device_map_strategy'].to_list(),
                old_results_df['num_jobs'].to_list()
            ))

            print(f'Resuming benchmark with {len(completed_runs)} runs complete.')

        else:
            print(f'No data to resume from, starting from scratch.')
            completed_runs = []

    # If we are not resuming an old run, empty datafile if it exists
    # and start with empty list for completed runs.
    else:
        # Initialize and save empty results object
        results = Results(results_dir)
        results.save_result(overwrite = True)
        completed_runs = []

    # Build parameter sets, excluding any which are already complete
    parameter_sets = []

    for device_map_strategy in device_map_strategies:

        # Need to handle CPU jobs differently than GPU jobs, they
        # use different device maps and job numbers.
        if 'CPU' in device_map_strategy.split(' '):
            for jobs in num_CPU_jobs:
                parameter_set = (device_map_strategy, jobs)

                # Note: the device map/job number loop produces a condition
                # where we call for 20 jobs on physical cores only, don't add
                # this condition to the list because we are only working with
                # 10 physical cores
                if (parameter_set not in completed_runs) and (parameter_set != ('CPU physical cores only', 20)):
                    parameter_sets.append(parameter_set)

        elif 'GPU' in device_map_strategy.split(' '):
            for jobs in num_GPU_jobs:
                parameter_set = (device_map_strategy, jobs)

                if parameter_set not in completed_runs:
                    parameter_sets.append(parameter_set)

    # Loop on parameter sets to run jobs
    for parameter_set in parameter_sets:

        # Split out the parameters for this run
        run_device_map_strategy = parameter_set[0]
        run_jobs = parameter_set[1]

        # Figure out how many abstracts we need to give each worker process
        run_abstracts = num_abstracts // run_jobs

        # Give torch CPU threads based on device map for this run, if appropriate
        if 'CPU' in run_device_map_strategy.split(' '):
            if run_device_map_strategy == 'CPU physical cores only':
                torch.set_num_threads(10 // run_jobs)

            elif run_device_map_strategy == 'CPU only hyperthreading':
                torch.set_num_threads(20 // run_jobs)

        # If this is a GPU run, start a counter to pick GPUs for jobs
        if 'GPU' in run_device_map_strategy.split(' '):
            gpu_index = 0

        # If this run is not using GPU(s), pass none for GPU related parameters
        else:
            gpu_index = None
            gpu = None

        # Instantiate pool with one member for each job we need to run
        pool = mp.Pool(
            processes = run_jobs,
            maxtasksperchild = 1
        )

        # Make results object for run
        results = Results(results_dir)
        results.data['device_map_strategy'].append(run_device_map_strategy)
        results.data['num_jobs'].append(run_jobs)

        print(f'\nStarting benchmark with {run_jobs} concurrent jobs and {run_abstracts} abstracts per job using {run_device_map_strategy}.')

        # Start timer
        start = time.time()

        # Loop on jobs for this run
        for i in list(range(0, run_jobs)):

            # Pick GPU for run, if needed
            if 'GPU' in run_device_map_strategy.split(' '):
                gpu = gpus[gpu_index]
                print(f'Job {i}: using GPU {gpu}')

            result = pool.apply_async(start_job,
                args = (i, db_name, user, passwd, host, run_abstracts, run_device_map_strategy, gpu_index, gpu),
                callback = collect_result
            )

            # Increment GPU index if needed - we have four GPUs, so when the index gets 
            # to 3 (0 anchored), reset it back to 0, otherwise increment it.
            if 'GPU' in run_device_map_strategy.split(' '):
                if gpu_index == 3:
                    gpu_index = 0
                
                else:
                    gpu_index += 1

        # Clean up
        pool.close()
        pool.join()

        # Stop the timer and log the result
        dT = time.time() - start
        results.data['summarization_time'].append(dT)
        results.save_result()


def start_job(i, db_name, user, passwd, host, num_abstracts, device_map_strategy, gpu_index, gpu):
    
    print(f'Job {i}: starting.')

    # Assign job to GPU if needed
    if 'GPU' in device_map_strategy.split(' '):
        torch.cuda.set_device(gpu_index)
        print(f'Job {i} has GPU {gpu_index}: {gpu}')
    
    # Fire up the model for this run
    model, tokenizer, gen_cfg = start_llm(device_map_strategy, gpu)

    # Get an abstract to summarize
    rows = get_rows(db_name, user, passwd, host, num_abstracts)

    row_count = 1

    for row in rows:

        # Get abstract text for this row
        abstract = row[1]

        # Make sure this abstract actually has content to be summarized
        if abstract != None:

            # Do the summary
            summary = summarize(abstract, model, tokenizer, gen_cfg, device_map_strategy)
            print(f'Job {i}: finished abstract {row_count}.')

        else:
            # Print a warning if the abstract we pulled is empty
            print(f'Job {i}: abstract {row_count} empty.')

        row_count += 1

    print(f'Job {1}: done.')

    return True

def get_rows(db_name, user, passwd, host, num_abstracts):
        
    # Open connection to PUBMED database on postgreSQL server, create connection
    connection = psycopg2.connect(f'dbname={db_name} user={user} password={passwd} host={host}')

    # Start new reader cursor
    read_cursor = connection.cursor()

    # Loop until we have num_abstracts non-empty rows to return. Note: ideally we would go back to
    # the article parsing script and not put empty abstracts into the SQL database. Let's do
    # that later, but this will work for now to get us were we want to go. Also, this is not
    # being timed as part of the benchmark, so any inefficacy in selecting a few hundred abstracts
    # is irrelevant

    # Get 2x the number of rows we want
    read_cursor.execute('SELECT * FROM abstracts ORDER BY random() LIMIT %s', (num_abstracts*2,))

    # Collect non-empty rows until we have enough
    rows = []

    for row in read_cursor:

        abstract = row[1]

        if abstract != None:
            rows.append(row)

        if len(rows) == num_abstracts:
            break

    read_cursor.close()

    return rows


def start_llm(device_map_strategy, gpu):
        
        # Set device_map parameter value for Huggingface

        if gpu == None:
            device_map = 'cpu'

        else:
            device_map = gpu

        # Initialize model with selected device map
        model = AutoModelForSeq2SeqLM.from_pretrained(
            "haining/scientific_abstract_simplification", 
            device_map = device_map
        )
        
        # Load generation config from model and set some parameters as desired
        gen_cfg = GenerationConfig.from_model_config(model.config)
        gen_cfg.max_length = 256
        gen_cfg.top_p = 0.9
        gen_cfg.do_sample = True

        # Initialize the tokenizer
        tokenizer = AutoTokenizer.from_pretrained("haining/scientific_abstract_simplification")

        return model, tokenizer, gen_cfg

def summarize(abstract, model, tokenizer, gen_cfg, device_map_strategy):
        
    # Prepend the prompt to this abstract and encode
    encoding = tokenizer(
        'summarize, simplify, and contextualize: ' + abstract, 
        max_length = 672, 
        padding = 'max_length', 
        truncation = True, 
        return_tensors = 'pt'
    )

    # Move to GPU if appropriate
    if 'CPU' not in device_map_strategy.split(' '):
        encoding = encoding.to('cuda')
    
    # Generate summary
    decoded_ids = model.generate(
        input_ids = encoding['input_ids'],
        attention_mask = encoding['attention_mask'], 
        generation_config = gen_cfg
    )
    
    # Decode summary
    summary = tokenizer.decode(decoded_ids[0], skip_special_tokens = True)

    return summary

def collect_result(result):
    # Need a dummy return here since we don't have good logging setup
    # this ensures that anything that a worker process sends to STDOUT
    # or STDERR actually shows up in the terminal
    return True

class Results:
    '''Class to hold objects and methods for
    collection of results'''

    def __init__(self, results_dir):

        # Output file for results
        self.output_file = f'{results_dir}/results.csv'

        # Independent vars for run
        self.data = {}
        self.data['num_jobs'] = []
        self.data['device_map_strategy'] = []
        self.data['summarization_time'] = []

    def save_result(self, overwrite = False):

        # Make dataframe of new results
        results_df = pd.DataFrame(self.data)

        # Read existing results if any and concatenate new results if desired
        if overwrite == False:
            if os.path.exists(self.output_file):
                old_results_df = pd.read_csv(self.output_file)
                results_df = pd.concat([old_results_df, results_df])

        else:
            print('Clearing any old results.')

        # Save results for run to csv
        results_df.to_csv(self.output_file, index = False)