import time
import psycopg2
import psycopg2.extras
# import torch
# import datetime
# import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig

import config as conf

if __name__ == "__main__":

    # Open connection to PUBMED database on postgreSQL server, create connection
    con = psycopg2.connect(f'dbname={conf.DB_NAME} user={conf.USER} password={conf.PASSWD} host={conf.HOST}')
    cur = con.cursor()

    # Next we need a table for the abstract summaries. First check to see if it exists already by 
    # trying to select it
    cur.execute('select exists(select * from information_schema.tables where table_name=%s)', ('abstract_summaries',))
    data_exists = cur.fetchone()[0]

    # If we get nothing back, we need to create the table
    if data_exists == False:
        print('\nCreating SQL table with PMC ID and abstract summary\n')

        cur.execute('''
            CREATE TABLE IF NOT EXISTS abstract_summaries(
            pmc_id varchar(12), abstract_summary text)
        ''')

        con.commit()

    # If the abstract summary table exists, make sure it's empty to start the run
    else:
        print('\nAbstract summary table exists\n')
        cur.execute('TRUNCATE abstract_summaries RESTART IDENTITY')
        con.commit()

    # Create a second cursor for reading and looping on chunks of rows from the database
    cur2 = con.cursor()

    # Get n rows from table of abstracts to be summarized
    n = 10
    cur2.execute('SELECT * FROM abstracts LIMIT %s', (n,))

    # Initialize LLM for summarization
    model = AutoModelForSeq2SeqLM.from_pretrained("haining/scientific_abstract_simplification")
    
    # Load generation config from model and set some parameters as desired
    gen_cfg = GenerationConfig.from_model_config(model.config)
    gen_cfg.max_length = 256
    gen_cfg.top_p = 0.9
    gen_cfg.do_sample = True

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("haining/scientific_abstract_simplification")

    # Set prompt to prepend to abstracts
    instruction = "summarize, simplify, and contextualize: "

    # Set up to time the different parts of the loop.
    row_count = 0
    summarization_time = 0
    insert_time = 0

    # Start loop timer
    loop_start = time.time()

    # Loop on abstract rows
    for row in cur2:
        # Make sure this abstract actually has content to be summarized
        if row[1] != None:

            row_count += 1
            print(f'Summarizing abstract {row_count}', end = '\r')

            # Start summarization timer
            summarization_start = time.time()

            # Encode prompt and abstract
            encoding = tokenizer(
                instruction + row[1], 
                max_length = 672, 
                padding = 'max_length', 
                truncation = True, 
                return_tensors = 'pt'
            )
            
            # Generate summary
            decoded_ids = model.generate(
                input_ids = encoding['input_ids'],
                attention_mask = encoding['attention_mask'], 
                generation_config = gen_cfg
            )
            
            # Decode summary
            summary = tokenizer.decode(decoded_ids[0], skip_special_tokens = True)
            
            # Stop summarization timer and add dT to total summarization time
            summarization_end = time.time()
            summarization_time += summarization_end - summarization_start

            # Start summary table insert timer
            insert_start = time.time()

            # Insert the new summary into the table
            cur.execute("INSERT INTO abstract_summaries (pmc_id, abstract_summary) VALUES(%s, %s)", (row[0], summary))
            con.commit()

            # Stop the insert timer and add dTt to total summarization time
            insert_end = time.time()
            insert_time += insert_end - insert_start

    # Stop the loop timer
    loop_end = time.time()

    # Get total time spent in loop and calculate loading time as
    # difference between total time in loop and time spent summarizing
    # and inserting
    total_time = loop_end - loop_start
    loading_time = total_time - summarization_time - insert_time

    # Results
    print(f'Total time: {total_time}')
    print(f'Summarization time: {round(summarization_time, 1)}')
    print(f'Insert time: {round(insert_time, 3)}')
    print(f'Loading time: {round(loading_time, 4)}\n')

    print(f'Mean summarization time: {round((summarization_time / n), 1)}')
    print(f'Mean insert time: {round((insert_time / n), 3)}')
    print(f'Mean loading time: {round((loading_time / n), 4)}\n')