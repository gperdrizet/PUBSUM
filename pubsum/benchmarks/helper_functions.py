import os
import psycopg2
import transformers
import pandas as pd

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

            for unique_independent_var in unique_independent_vars:
                old_results.append(old_results_df[unique_independent_var].to_list())

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

def summarize(
    abstracts: [str], 
    model: transformers.T5ForConditionalGeneration, 
    tokenizer: transformers.T5TokenizerFast, 
    gen_cfg: transformers.GenerationConfig,
    use_GPU: bool = False
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

    # Move to GPU, if needed
    if use_GPU == True:
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
