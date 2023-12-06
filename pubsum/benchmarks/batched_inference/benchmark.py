import os
import gc
import pandas as pd
import psycopg2
import time
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig, BitsAndBytesConfig, GPTQConfig

def benchmark(db_name, user, passwd, host, resume, results_dir, num_abstracts, replicates, batch_sizes):
    
    print(f'\nRunning batched inference benchmark. Resume = {resume}.\n')

    # If we are resuming a prior run, read old data and collect the
    # completed conditions as a list of lists so we can skip them
    if resume == 'True':

        # Read existing results if any
        if os.path.exists(f'{results_dir}/results.csv'):

            old_results_df = pd.read_csv(f'{results_dir}/results.csv')

            completed_runs = list(zip(
                old_results_df['replicate'].to_list(),
                old_results_df['batch size'].to_list()
            ))

            print(f'Resuming benchmark with {len(completed_runs)} runs complete.')

        else:
            print(f'No data to resume from, starting from scratch.')
            completed_runs = []

    # If we are not resuming an old run, empty datafile if it exists
    else:
        # Initialize and save empty results object
        results = Results(results_dir)
        results.save_result(overwrite = True)

        # Set completed runs to empty list
        completed_runs = []

    for batch_size in batch_sizes:
        for replicate in range(replicates):

            run_tuple = (replicate, batch_size)

            if run_tuple not in completed_runs:

                print(f'Starting benchmark run on {num_abstracts} abstracts with batch size {batch_size}, replicate {replicate}.')

                # Fire up the model for this run
                model, tokenizer, gen_cfg = start_llm()
                model_memory_footprint = model.get_memory_footprint()

                # Get rows from abstracts table
                rows = get_rows(db_name, user, passwd, host, num_abstracts)

                batch_count = 0

                # Instantiate results object for this run
                results = Results(results_dir)

                # Collect run parameters to results
                results.data['abstracts'].append(num_abstracts)
                results.data['replicate'].append(replicate)
                results.data['batch size'].append(batch_size)
                results.data['model GPU memory footprint (bytes)'].append(model_memory_footprint)

                # Star the clock
                summarization_start = time.time()

                for batch in batches(rows, batch_size):

                    batch_count += 1

                    print(f'Summarizing batch {batch_count} of {num_abstracts // batch_size}.')

                    # Get PMCIDs and abstract texts for this batch
                    pmcids = [row[0] for row in batch]
                    abstracts = [row[1] for row in batch]

                    # Do the summary
                    summary = summarize(abstracts, model, tokenizer, gen_cfg)

                # Stop the clock
                dT = time.time() - summarization_start

                # Get max memory used and reset
                max_memory = torch.cuda.max_memory_allocated()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                # Collect data
                results.data['max memory allocated (bytes)'].append(max_memory)
                results.data['summarization time (sec.)'].append(dT)
                results.data['summarization rate (abstracts/sec.)'].append(num_abstracts/dT)

                # Save the result
                results.save_result()

            # Get rid of model and tokenizer from run, free up memory
            del model
            del tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            print('Done.\n')

    print()

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


def start_llm():
        
        # Place model on single GPU
        device_map = 'cuda:0'

        # Set quantization config
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_compute_dtype=torch.float16
        )

        # Initialize the tokenizer
        tokenizer = AutoTokenizer.from_pretrained('haining/scientific_abstract_simplification')

        # Initialize model
        model = AutoModelForSeq2SeqLM.from_pretrained(
            'haining/scientific_abstract_simplification', 
            device_map=device_map,
            quantization_config=quantization_config
        )

        model.to_bettertransformer()
        
        # Load generation config from model and set some parameters as desired
        gen_cfg = GenerationConfig.from_model_config(model.config)
        gen_cfg.max_length = 256
        gen_cfg.top_p = 0.9
        gen_cfg.do_sample = True

        return model, tokenizer, gen_cfg

def summarize(abstracts, model, tokenizer, gen_cfg):
        
    # Prepend the prompt to this abstract and encode
    inputs = ['summarize, simplify, and contextualize: ' + abstract for abstract in abstracts]

    encoding = tokenizer(
        inputs, 
        max_length = 672, 
        padding = 'max_length', 
        truncation = True, 
        return_tensors = 'pt'
    )

    # Move to GPU
    encoding = encoding.to('cuda')
    
    # Generate summary
    decoded_ids = model.generate(
        input_ids = encoding['input_ids'],
        attention_mask = encoding['attention_mask'], 
        generation_config = gen_cfg
    )
    
    # Decode summaries
    summaries = tokenizer.decode(decoded_ids[0], skip_special_tokens = True)

    return summaries

class Results:
    '''Class to hold objects and methods for
    collection of results'''

    def __init__(self, results_dir):

        # Output file for results
        self.output_file = f'{results_dir}/results.csv'

        # Independent vars for run
        self.data = {}
        self.data['abstracts'] = []
        self.data['replicate'] = []
        self.data['batch size'] = []
        self.data['summarization time (sec.)'] = []
        self.data['summarization rate (abstracts/sec.)'] = []
        self.data['model GPU memory footprint (bytes)'] = []
        self.data['max memory allocated (bytes)'] = []

    def save_result(self, overwrite = False):

        # Make dataframe of new results
        results_df = pd.DataFrame(self.data)

        if overwrite == False:

            # Read existing results if any and concatenate new results
            if os.path.exists(self.output_file):
                old_results_df = pd.read_csv(self.output_file)
                results_df = pd.concat([old_results_df, results_df])

        else:
            print('Clearing any old results.')

        # Save results for run to csv
        results_df.to_csv(self.output_file, index = False)

def batches(lst, n):
    '''Yield successive n-sized chunks from lst.'''
    for i in range(0, len(lst), n):
        yield lst[i:i + n]