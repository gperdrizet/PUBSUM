'''Initial execution time benchmark of first working loop 
to load abstracts from the SQL database, summarize them w
ith the LLM and then insert the summaries into a new SQL table.
'''

import time
import os
import psycopg2
import psycopg2.extras
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig

def benchmark(db_name, user, passwd, host, resume, results_dir, num_abstracts, replicates):

    print(f'\nRunning naive load, summarize, insert execution time benchmark. Resume = {resume}\n')

    # Connect to postgresql server
    print('Connecting to SQL server.')
    connection = psycopg2.connect(f'dbname={db_name} user={user} password={passwd} host={host}')
    write_cursor = connection.cursor()

    # Next we need a table to insert into. Make a scratch table just for this test.
    # If the table exists already, delete it.
    print('Creating target table for summaries.')
    write_cursor.execute('''
        DROP TABLE IF EXISTS summary_benchmark
    ''')

    connection.commit()

    # Create file path table
    write_cursor.execute('''
        CREATE TABLE IF NOT EXISTS summary_benchmark
        (pmc_id varchar(12), abstract_summary text)
    ''')

    connection.commit()

    write_cursor.close()

    # Initialize model and set-up generation config
    print('Initializing model and tokenizer.')
    model = AutoModelForSeq2SeqLM.from_pretrained('haining/scientific_abstract_simplification')
    
    gen_cfg = GenerationConfig.from_model_config(model.config)
    gen_cfg.max_length = 512
    gen_cfg.top_p = 0.9
    gen_cfg.do_sample = True

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('haining/scientific_abstract_simplification')

    # Prompt to prepend to abstracts
    INSTRUCTION = 'summarize, simplify, and contextualize: '

    # If we are resuming a prior run, read old data and find where we left
    # off, get the last completed replicate and start the replicate loop
    # counter there
    if resume == 'True':

        # Read existing results if any
        if os.path.exists(f'{results_dir}/results.csv'):

            old_results_df = pd.read_csv(f'{results_dir}/results.csv')

            # Get last replicate num, use that to initialize the replicate counter
            # to start at the next replicate
            last_complete_replicate = old_results_df['replicate'].to_list()[-1]
            replicate = last_complete_replicate + 1

            # Check if we already completed the run or not by comparing
            # the replicate number we just loaded to the requested 
            # number of replicates

            if replicate > replicates:
                print('Run was complete.')

            else:
                print(f'Resuming benchmark at replicate {replicate}.')

        else:
            print('No data to resume from, starting at replicate 1.')
            replicate = 1

    # If we are not resuming an old run, empty datafile if it 
    # exists and start replicate count at 1
    else:
        # Initialize and save empty results object
        results = Results(results_dir)
        results.save_result(overwrite = True)
        replicate = 1

    # Loop on replicates
    while replicate <= replicates:

        # Create result collector for this replicate
        results = Results(results_dir)

        # Create a cursor for reading and get n random rows
        read_cursor = connection.cursor()
        read_cursor.execute('SELECT * FROM abstracts ORDER BY random() LIMIT %s', (num_abstracts,))

        # Create a second cursor for writing
        write_cursor = connection.cursor()

        # Loop on abstracts in this replicate and time everything
        row_count = 0
        replicate_start = time.time()
        total_summarization_time = 0
        total_insert_time = 0

        for row in read_cursor:

            row_count += 1

            # Parse row into article accession number and abstract text
            pmcid = row[0]
            abstract = row[1]

            # If the abstract is empty, skip it
            if abstract != None:

                print(f'Replicate {replicate}, row: {row_count}')

                # Time the LLM doing it's thing
                summarization_start = time.time()

                # Add the prompt to the abstract and tokenize
                encoding = tokenizer(
                    INSTRUCTION + abstract, 
                    max_length = 672, 
                    padding = 'max_length', 
                    truncation = True, 
                    return_tensors = 'pt'
                )
                
                # Generate the summary
                decoded_ids = model.generate(
                    input_ids = encoding['input_ids'],
                    attention_mask = encoding['attention_mask'], 
                    generation_config = gen_cfg
                )
                
                # Decode summary tokens back to text
                summary = tokenizer.decode(decoded_ids[0], skip_special_tokens = True)
                
                # Stop the summarization timer and add the dT to the total
                summarization_end = time.time()
                total_summarization_time += summarization_end - summarization_start

                # Start the SQL insert timer
                insert_start = time.time()

                # Insert the summary text into the SQL table
                write_cursor.execute("INSERT INTO summary_benchmark (pmc_id, abstract_summary) VALUES(%s, %s)", (pmcid, summary))
                connection.commit()

                # Stop the insert timer and add the dT to the total
                insert_end = time.time()
                total_insert_time += insert_end - insert_start

            else:
                print('Empty abstract.')

        # Stop iteration timer
        replicate_end = time.time()

        # Calculate total time to complete this replicate
        total_replicate_time = replicate_end - replicate_start

        # Calculate average time to complete one abstract in this replicate
        mean_total_time = total_replicate_time / row_count

        # Calculate average summarization time
        mean_summarization_time = total_summarization_time / row_count

        # Calculate mean insert time
        mean_insert_time = total_insert_time / row_count

        # Calculate total load time
        total_loading_time = total_replicate_time - total_summarization_time - total_insert_time

        # Calculate mean load time
        mean_loading_time = total_loading_time / row_count

        # Collect data from this replicate
        results.data['replicate'].append(replicate)
        results.data['num_abstracts'].append(row_count)
        results.data['total_replicate_time'].append(total_replicate_time)
        results.data['total_summarization_time'].append(total_summarization_time)
        results.data['total_insert_time'].append(total_insert_time)
        results.data['total_loading_time'].append(total_loading_time)
        results.data['mean_total_time'].append(mean_total_time)
        results.data['mean_summarization_time'].append(mean_summarization_time)
        results.data['mean_insert_time'].append(mean_insert_time)
        results.data['mean_loading_time'].append(mean_loading_time)

        # Save the results from this replicate
        results.save_result()

        # Close our cursors for next replicate
        read_cursor.close()
        write_cursor.close()

        replicate += 1

        print('Done.')

class Results:
    '''Class to hold objects and methods for
    collection of results'''

    def __init__(self, results_dir):

        # Output file for results
        self.output_file = f'{results_dir}/results.csv'

        # Independent vars for run
        self.data = {}
        self.data['replicate'] = []
        self.data['num_abstracts'] = []
        self.data['total_replicate_time'] = []
        self.data['total_summarization_time'] = []
        self.data['total_insert_time'] = []
        self.data['total_loading_time'] = []
        self.data['mean_total_time'] = []
        self.data['mean_summarization_time'] = []
        self.data['mean_insert_time'] = []
        self.data['mean_loading_time'] = []

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