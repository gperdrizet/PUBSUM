import os
import psycopg2
import transformers
import pandas as pd
from typing import List

def get_rows(
    db_name: str, 
    user: str, 
    passwd: str, 
    host: str, 
    num_abstracts: int
) -> List[List[str]]:
        
    # Open connection to PUBMED database on postgreSQL server, create connection
    connection = psycopg2.connect(f'dbname={db_name} user={user} password={passwd} host={host}')

    # Start new reader cursor
    read_cursor = connection.cursor()

    # Loop until we have num_abstracts non-empty rows to return. Note: ideally we would go back to
    # the article parsing script and not put empty abstracts into the SQL database. Let's do
    # that later, but this will work for now to get us were we want to go. Also, this is not
    # being timed as part of the benchmark, so any inefficacy is irrelevant to the results

    # On the first loop, the number of abstract to get is the total number we want
    abstracts_remaining = num_abstracts

    # Collector for result
    rows = []

    # Loop until we have enough abstracts
    while abstracts_remaining > 0:

        # Get abstracts
        read_cursor.execute('SELECT * FROM abstracts ORDER BY random() LIMIT %s', (abstracts_remaining,))

        # Loop on the returned abstracts
        for row in read_cursor:

            # Get the abstract text
            abstract = row[1]

            # If we got a non-empty abstract, add the row to the result
            if abstract != None:
                rows.append(row)

        # Update abstracts remaining
        abstracts_remaining = num_abstracts - len(rows)
        
    read_cursor.close()

    return rows

def summarize(
    abstracts: List[str], 
    model: transformers.T5ForConditionalGeneration, 
    tokenizer: transformers.T5TokenizerFast, 
    gen_cfg: transformers.GenerationConfig,
    use_GPU: bool = False
) -> List[str]:
        
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

    '''Class to hold objects and methods for collection of results.'''

    def __init__(
        self,
        independent_vars: List[str],
        dependent_vars: List[str],
        output_dir: str,  
        output_filename: str = 'results.csv',
        resume: bool = False,
    ):

        # Add variable keys
        self.independent_vars = independent_vars
        self.dependent_vars = dependent_vars

        # Output file for results
        self.output_file = f'{output_dir}/{output_filename}'

        # Create dict of empty lists for data collection
        self.data = {}
        for var in self.independent_vars + self.dependent_vars:
            self.data[var] = []

        # Resume data collection from disk only if asked
        if resume == False:

            # Empty list for tuples of values of independent variables for
            # completed runs, used to make sure we are not repeating runs
            self.completed_runs = []

        elif resume == True:

            # Read existing results if any
            if os.path.exists(self.output_file):
                old_results_df = pd.read_csv(self.output_file)

                # Loop on old results and collect values of independent vars
                old_results = []
                for var in self.independent_vars:
                    old_results.append(old_results_df[var].to_list())

                # Zip the lists of old data into tuples
                self.completed_runs = list(zip(*old_results))
                print(f'Resuming data collection with {len(self.completed_runs)} runs complete')

            # If we don't have data to resume from, call it a fresh start
            else:
                print(f'No data to resume from, starting from scratch')
                self.completed_runs = []

    def save_result(self, overwrite: bool = False) -> None:

        '''Persist current contents of self.data to disk. Overwrite 
        any data in self.output_file by default or append if requested.'''

        # Make dataframe of new results
        results_df = pd.DataFrame(self.data)

        if overwrite == False:

            # Read existing results, if any and concatenate new results
            if os.path.exists(self.output_file):
                old_results_df = pd.read_csv(self.output_file)
                results_df = pd.concat([old_results_df, results_df])

        else:
            print('Clearing any old results')

        # Save results for run to csv
        results_df.to_csv(self.output_file, index = False)

    def clear_results(self) -> None:

        '''Empties contents of self.data.'''

        # Assign empty list for all collection vars
        for collection_var in self.collection_vars:
            self.data[collection_var] = []