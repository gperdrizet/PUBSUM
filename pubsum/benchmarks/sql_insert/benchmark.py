import os
import gc
import time
import psycopg2
import psycopg2.extras as extras
import itertools
import pandas as pd
from io import StringIO
from typing import List

def benchmark(
    helper_funcs,
    resume: bool, 
    master_file_list: str,
    results_dir: str,
    abstract_nums: int,
    insert_strategies: List[str],
    replicates: int, 
    db_name: str, 
    user: str, 
    passwd: str, 
    host: str, 
):

    print(f'\nRunning SQL insert benchmark. Resume = {resume}.')

    # Set list of keys for the data we want to collect
    collection_vars = [
        'abstracts',
        'insert time (sec.)',
        'insert rate (abstracts/sec.)',
        'insert strategy',
        'replicate'
    ]

    # Subset of independent vars which are sufficient to uniquely identify each run
    unique_collection_vars = [
        'insert strategy',
        'abstracts',
        'replicate'
    ]

    # Handel resume request by reading or emptying old data, as appropriate
    completed_runs = helper_funcs.resume_run(
        resume=resume, 
        results_dir=results_dir,
        collection_vars=collection_vars,
        unique_collection_vars=unique_collection_vars
    )

    # Construct parameter sets
    replicate_numbers = list(range(1, replicates + 1))

    parameter_sets = itertools.product(
        insert_strategies,
        abstract_nums,
        replicate_numbers
    )

    if len(insert_strategies) * len(abstract_nums) * len(replicate_numbers) == len(completed_runs):
        print('Run was complete.')

    else:

        # Read file list into pandas dataframe
        print('Reading file list into pandas df.')
        file_list_df = pd.read_csv(master_file_list)

        # Extract article paths and PMC IDs
        print('Extracting PMC ID and file path.')
        article_paths_df = pd.DataFrame(file_list_df, columns=['AccessionID', 'Article File'])

        # Connect to postgresql server
        print('Connecting to SQL server.')
        connection = psycopg2.connect(f'dbname={db_name} user={user} password={passwd} host={host}')

        # Loop on parameter sets
        for parameter_set in parameter_sets:

            # Check if we have already completed this parameter set
            if parameter_set not in completed_runs:

                # Unpack parameters from set
                insert_strategy, num_abstracts, replicate = parameter_set

                print(f'\nSQL insert strategy benchmark:\n')
                print(f' Replicate: {replicate} of {replicates}')
                print(f' Insert strategy {insert_strategy}')
                print(f' Abstracts: {num_abstracts}')

                # Instantiate results object for this run
                results = helper_funcs.Results(
                    results_dir=results_dir,
                    collection_vars=collection_vars
                )

                # Initialize a fresh write cursor and target table
                write_cursor = make_target_table_write_cursor(connection=connection)

                # Randomly pick n rows of abstracts
                rows = article_paths_df.sample(num_abstracts)

                # Do and time the insert
                insert_time = insert(
                    rows=rows, 
                    insert_strategy=insert_strategy, 
                    write_cursor=write_cursor, 
                    connection=connection
                )

                # Collect result
                results.data['abstracts'].append(num_abstracts)
                results.data['insert time (sec.)'].append(insert_time)
                results.data['insert rate (abstracts/sec.)'].append(num_abstracts/insert_time)
                results.data['insert strategy'].append(insert_strategy)
                results.data['replicate'].append(replicate)

                # Save result
                results.save_result()

                # Clean up
                write_cursor.close()

                print(' Done.')

        # Clean up
        write_cursor = connection.cursor()

        write_cursor.execute('''
            DROP TABLE IF EXISTS insert_benchmark
        ''')

        connection.commit()

        write_cursor.close()
        connection.close()

    return True

def make_target_table_write_cursor(
    connection: psycopg2.extensions.connection
) -> psycopg2.extensions.cursor:

    write_cursor = connection.cursor()
    
    # Next we need a table to insert into. Make a scratch table just for this test.
    # If the table exists already, delete it.
    write_cursor.execute('''
        DROP TABLE IF EXISTS insert_benchmark
    ''')

    connection.commit()

    # Then create it
    write_cursor.execute('''
        CREATE TABLE IF NOT EXISTS insert_benchmark(
        pmc_id varchar(12), article_file_path varchar(29))
    ''')
    
    connection.commit()

    return write_cursor

def insert(
    rows: int, 
    insert_strategy: str, 
    write_cursor: psycopg2.extensions.cursor, 
    connection: psycopg2.extensions.cursor
) -> float:

    if insert_strategy == 'execute_many':

        start_time = time.time()

        write_cursor.executemany(
            '''INSERT INTO insert_benchmark(pmc_id, article_file_path) VALUES(%s, %s)''', 
            zip(rows['AccessionID'].to_list(), rows['Article File'].to_list())
        )

        connection.commit()

    if insert_strategy == 'execute_batch':

        start_time = time.time()

        extras.execute_batch(
            write_cursor, 
            '''INSERT INTO insert_benchmark(pmc_id, article_file_path) VALUES(%s, %s)''', 
            zip(rows['AccessionID'].to_list(), rows['Article File'].to_list()), 
            page_size = len(rows)
        )

        connection.commit()

    if insert_strategy == 'execute_values':

        start_time = time.time()

        extras.execute_values(
            write_cursor,
            '''INSERT INTO insert_benchmark(pmc_id, article_file_path) VALUES %s''',
            zip(rows['AccessionID'].to_list(), rows['Article File'].to_list()),
            page_size = len(rows),
            template = '(%s, %s)'
        )

        connection.commit()

    if insert_strategy == 'mogrify':

        start_time = time.time()

        tuples = zip(rows['AccessionID'].to_list(), rows['Article File'].to_list())
        values = [write_cursor.mogrify("(%s,%s)", tup).decode('utf8') for tup in tuples]
        query  = "INSERT INTO insert_benchmark(pmc_id, article_file_path) VALUES " + ",".join(values)

        write_cursor.execute(query, tuples)
        connection.commit()

    if insert_strategy == 'stringIO':

        start_time = time.time()

        buffer = StringIO()
        rows.to_csv(buffer, index = False, header = False)
        buffer.seek(0)

        write_cursor.copy_from(buffer, 'insert_benchmark', sep = ",")
        connection.commit()
        
    return time.time() - start_time