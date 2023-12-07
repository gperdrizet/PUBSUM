import os
import gc
import pandas as pd
import psycopg2
import time
import torch
import itertools
import transformers

def benchmark(
    resume: str, 
    results_dir: str, 
    replicates: int, 
    batches: int, 
    batch_sizes: [int], 
    quantization_strategies: [str],
    db_name: str, 
    user: str, 
    passwd: str, 
    host: str
) -> bool:
    
    print(f'\nRunning batched inference benchmark. Resume = {resume}.\n')

    # Set list of keys for the data we want to collect
    independent_vars = [
        'abstracts',
        'replicate',
        'batches',
        'batch size',
        'summarization time (sec.)',
        'summarization rate (abstracts/sec.)',
        'model GPU memory footprint (bytes)',
        'max memory allocated (bytes)'
    ]

    # Subset of independent vars which are sufficient to uniquely identify each run
    unique_independent_vars = [
        'replicate', 
        'batch size'
    ]

    # Handel resume request by reading or emptying old data, as appropriate
    completed_runs = resume_run(
        resume=resume, 
        results_dir=results_dir,
        independent_vars=independent_vars,
        unique_independent_vars=unique_independent_vars
    )

    # Construct parameter sets
    replicate_numbers = list(range(1, replicates + 1))

    parameter_sets = itertools.product(
        replicate_numbers,
        batch_sizes
    )

    # Loop on parameter sets
    for parameter_set in parameter_sets:

        # Check if we have already completed this parameter set
        if parameter_set not in completed_runs:

            # Unpack parameters from set
            replicate = parameter_set[0]
            batch_size = parameter_set[1]

            # Calculate total abstracts needed for job
            num_abstracts = batches * batch_size

            print(f'\nBatched summarization benchmark:\n')
            print(f' Replicate: {replicate} of {replicates}')
            print(f' Batch size: {batch_size}')
            print(f' Batches: {batches}')
            print(f' Abstracts: {num_abstracts}\n')

            # Instantiate results object for this run
            results = Results(
                results_dir=results_dir,
                independent_vars=independent_vars
            )

            # Collect data for run parameters
            results.data['abstracts'].append(num_abstracts)
            results.data['replicate'].append(replicate)
            results.data['batches'].append(batches)
            results.data['batch size'].append(batch_size)

            # Fence to catch out of memory errors
            try:
                # Fire up the model for this run
                model, tokenizer, gen_cfg = start_llm()
                model_memory_footprint = model.get_memory_footprint()

                # Get enough rows to batch from abstracts table
                rows = get_rows(db_name, user, passwd, host, num_abstracts)

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
                    summaries = summarize(abstracts, model, tokenizer, gen_cfg)

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

                # Get rid of model and tokenizer from run, free up memory
                del model
                del tokenizer
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            # Catch out of memory errors
            except torch.cuda.OutOfMemoryError as oom:
                print(f'\n {oom}\n')

                # Since we failed on OOM, mark it in results
                results.data['model GPU memory footprint (bytes)'].append('OOM')
                results.data['max memory allocated (bytes)'].append('OOM')
                results.data['summarization time (sec.)'].append('OOM')
                results.data['summarization rate (abstracts/sec.)'].append('OOM')

            # Save the result
            results.save_result()
            print(' Done.')

    print()

    return True

def get_rows(
    db_name: str, 
    user: str, 
    passwd: str, 
    host: str, 
    num_abstracts: int
) -> [[str]]:
        
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


def start_llm() -> (
    transformers.T5ForConditionalGeneration, transformers.T5TokenizerFast, transformers.GenerationConfig
):
        
        # Place model on single GPU
        device_map = 'cuda:0'

        # Set quantization config
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

def summarize(
    abstracts: [str], 
    model: transformers.T5ForConditionalGeneration, 
    tokenizer: transformers.T5TokenizerFast, 
    gen_cfg: transformers.GenerationConfig
) -> [str]:
        
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

    def __init__(self, results_dir: str, independent_vars: [str]):

        # Output file for results
        self.output_file = f'{results_dir}/results.csv'

        # Create dict for data
        self.data = {}

        # Empty list to data dict for each independent var
        for independent_var in independent_vars:
            self.data[independent_var] = []

    def save_result(self, overwrite: bool = False) -> None:

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


def resume_run(
    resume: str, 
    results_dir: str, 
    independent_vars: [str], 
    unique_independent_vars: [str]
) -> list:

    '''Reads old data and collects the completed conditions 
    as a list of lists so we can skip them'''

    # Check if we are resuming
    if resume == 'True':

        # Read existing results if any
        if os.path.exists(f'{results_dir}/results.csv'):

            old_results_df = pd.read_csv(f'{results_dir}/results.csv')

            # Make a list of lists containing the data the requested
            # columns in the old results dataframe
            old_results = []

            for independent_var in independent_vars:
                old_results.append(old_results_df[independent_var].to_list())

            # Zip the lists of data into tuples
            completed_runs = list(zip(*old_results))

            print(f'Resuming benchmark with {len(completed_runs)} runs complete.')

        # If we don't have data to resume from, call it a fresh start
        else:
            print(f'No data to resume from, starting from scratch.')
            completed_runs = []

    # If we are not resuming a previous run...
    else:

        # Initialize and save empty results object
        results = Results(
            results_dir=results_dir,
            independent_vars=independent_vars
        )

        results.save_result(overwrite = True)

        # Set completed runs to empty list
        completed_runs = []

    return completed_runs