import os
import pandas as pd
import psycopg2
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig

def benchmark(db_name, user, passwd, host, resume, results_dir, num_abstracts, device_map_strategies):
    
    print(f'\nRunning huggingface device map benchmark. Resume = {resume}\n')

    # If we are resuming a prior run, read old data and collect the
    # completed conditions as a list of lists so we can skip them
    if resume == 'True':

        # Read existing results if any
        if os.path.exists(f'{results_dir}/results.csv'):

            old_results_df = pd.read_csv(f'{results_dir}/results.csv')

            completed_runs = list(zip(
                old_results_df['abstract'].to_list(),
                old_results_df['device map strategy'].to_list()
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


    for device_map_strategy in device_map_strategies:

        print(f'Starting benchmark run on {num_abstracts} abstracts with device map strategy {device_map_strategy}.')

        # Fire up the model for this run
        model, tokenizer, gen_cfg = start_llm(device_map_strategy)

        # Get rows from abstracts table
        rows = get_rows(db_name, user, passwd, host, num_abstracts)

        # Loop on rows
        row_count = 1

        for row in rows:

            run_tuple = (row_count, device_map_strategy)

            if run_tuple not in completed_runs:

                # Instantiate results object for this run
                results = Results(results_dir)

                # Get PMCID and abstract text for this row
                pmcid = row[0]
                abstract = row[1]

                # Make sure this abstract actually has content to be summarized
                if abstract != None:

                    # Collect run parameters to results
                    results.data['abstract'].append(row_count)
                    results.data['device map strategy'].append(device_map_strategy)
                    print(f'Summarizing {pmcid}: {row_count} of {num_abstracts}')

                    # Do and time the summary
                    summarization_start = time.time()
                    summary = summarize(abstract, model, tokenizer, gen_cfg, device_map_strategy)
                    dT = time.time() - summarization_start
                    results.data['summarization time (sec.)'].append(dT)
                    results.data['summarization rate (abstracts/sec.)'].append(1/dT)

                    # Save the result
                    results.save_result()

                else:
                    print(f'Empty abstract.')

            row_count += 1

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


def start_llm(device_map_strategy):
        
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

        model = AutoModelForSeq2SeqLM.from_pretrained("haining/scientific_abstract_simplification", device_map = device_map)
        
        # Load generation config from model and set some parameters as desired
        gen_cfg = GenerationConfig.from_model_config(model.config)
        gen_cfg.max_length = 256
        gen_cfg.top_p = 0.9
        gen_cfg.do_sample = True

        # Initialize the tokenizer
        tokenizer = AutoTokenizer.from_pretrained("haining/scientific_abstract_simplification")

        # Set prompt to prepend to abstracts
        #instruction = "summarize, simplify, and contextualize: "

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
    if device_map_strategy != 'CPU only':
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

class Results:
    '''Class to hold objects and methods for
    collection of results'''

    def __init__(self, results_dir):

        # Output file for results
        self.output_file = f'{results_dir}/results.csv'

        # Independent vars for run
        self.data = {}
        self.data['abstract'] = []
        self.data['device map strategy'] = []
        self.data['summarization time (sec.)'] = []
        self.data['summarization rate (abstracts/sec.)'] = []

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