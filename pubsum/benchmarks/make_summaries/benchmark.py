'''Initial benchmark of first working loop to load abstracts
from the SQL database, summarize them with the LLM and then
insert the summaries into a new SQL table. Goal with this
benchmark is to establish a time budget so we know how long
the load, summarize and insert operations are taking.'''

import time
import os
import psycopg2
import psycopg2.extras
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig

def benchmark(db_name, user, passwd, host, results_dir, num_abstracts, replicates):

    print('\nRunning naive load, summarize, insert execution time benchmark.\n')

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

    # Loop on replicates
    i = 1

    while i <= replicates:

        # Create result collector for this replicate
        results = Results(results_dir)

        # Create a cursor for reading and get n random rows
        read_cursor = connection.cursor()
        read_cursor.execute('SELECT * FROM abstracts ORDER BY random() LIMIT %s', (num_abstracts,))

        # Create a second cursor for writing
        write_cursor = connection.cursor()

        # Loop on abstracts in this replicate and time everything
        row_count = 1
        replicate_start = time.time()
        total_summarization_time = 0
        total_insert_time = 0

        for row in read_cursor:

            # Parse row into article accession number and abstract text
            pmcid = row[0]
            abstract = row[1]

            # If the abstract is empty, skip it
            if abstract != None:

                print(f'Rep {i}, row: {row_count}', end='\r')

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

            row_count += 1

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

        # Collect and save data from this replicate
        results.data['num_abstracts'].append(row_count)
        results.data['total_replicate_time'].append(total_replicate_time)
        results.data['total_summarization_time'].append(total_summarization_time)
        results.data['total_insert_time'].append(total_insert_time)
        results.data['total_loading_time'].append(total_loading_time)
        results.data['mean_total_time'].append(mean_total_time)
        results.data['mean_summarization_time'].append(mean_summarization_time)
        results.data['mean_insert_time'].append(mean_insert_time)
        results.data['mean_loading_time'].append(mean_loading_time)

        results.save_result()

        # Close our cursors for next replicate
        read_cursor.close()
        write_cursor.close()

        i += 1

        print()

class Results:
    '''Class to hold objects and methods for
    collection of results'''

    def __init__(self, results_dir):

        # Output file for results
        self.output_file = f'{results_dir}/results.csv'

        # Independent vars for run
        self.data = {}
        self.data['num_abstracts'] = []
        self.data['total_replicate_time'] = []
        self.data['total_summarization_time'] = []
        self.data['total_insert_time'] = []
        self.data['total_loading_time'] = []
        self.data['mean_total_time'] = []
        self.data['mean_summarization_time'] = []
        self.data['mean_insert_time'] = []
        self.data['mean_loading_time'] = []

    def save_result(self):

        # Make dataframe of new results
        results_df = pd.DataFrame(self.data)

        # Read existing results if any and concatenate new results
        if os.path.exists(self.output_file):
            old_results_df = pd.read_csv(self.output_file)
            results_df = pd.concat([old_results_df, results_df])

        # Save results for run to csv
        results_df.to_csv(self.output_file, index = False)