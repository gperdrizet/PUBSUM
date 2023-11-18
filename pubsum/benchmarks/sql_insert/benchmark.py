'''Benchmarking script to try out various methods
for inserting large numbers of rows into postgreSQL
tables via psycopg2
'''

import os
import time
import psycopg2
import psycopg2.extras as extras
import pandas as pd
from io import StringIO

def benchmark(master_file_list, db_name, user, passwd, host, results_dir, 
              num_abstracts, insert_strategies, num_replicates):

    print('\nRunning SQL insert benchmark.\n')

    # Read file list into pandas dataframe
    print('Reading file list into pandas df.')
    file_list_df = pd.read_csv(master_file_list)

    # Extract article paths and PMC IDs
    print('Extracting PMC ID and file path.')
    article_paths_df = pd.DataFrame(file_list_df, columns=['AccessionID', 'Article File'])

    # Connect to postgresql server
    print('Connecting to SQL server.')
    connection = psycopg2.connect(f'dbname={db_name} user={user} password={passwd} host={host}')

    # Loop on the insertion strategies
    for insert_strategy in insert_strategies:

        # Instantiate results object for this run
        results = Results(results_dir)

        # Loop on the abstract sample sizes
        for n in num_abstracts:

            print(f'Starting benchmark run on {n} abstract with insertion strategy {insert_strategy}.')

            # Do replicates for science
            i = 1
            while i <= num_replicates:

                print(f'Inserting replicate {i}', end = '\r')

                # Initialize a fresh write cursor and target table
                write_cursor = make_target_table_write_cursor(connection)

                # Randomly pick n rows of abstracts
                rows = article_paths_df.sample(n)

                # Do and time the insert
                insert_time = insert(rows, insert_strategy, write_cursor, connection)

                # Collect result
                results.data['num_abstracts'].append(n)
                results.data['insert_time'].append(insert_time)
                results.data['insert_strategy'].append(insert_strategy)
                results.data['replicate'].append(i)

                write_cursor.close()

                i += 1

            print()

        # Save result from run
        results.save_result()

    # Clean up
    write_cursor = connection.cursor()

    write_cursor.execute('''
        DROP TABLE IF EXISTS insert_benchmark
    ''')

    connection.commit()

    write_cursor.close()
    connection.close()


def make_target_table_write_cursor(connection):

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

def insert(rows, insert_strategy, write_cursor, connection):

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


class Results:
    '''Class to hold objects and methods for
    collection of results'''

    def __init__(self, results_dir):

        # Output file for results
        self.output_file = f'{results_dir}/results.csv'

        # Independent vars for run
        self.data = {}
        self.data['num_abstracts'] = []
        self.data['insert_time'] = []
        self.data['insert_strategy'] = []
        self.data['replicate'] = []

    def save_result(self):

        # Make dataframe of new results
        results_df = pd.DataFrame(self.data)

        # Read existing results if any and concatenate new results
        if os.path.exists(self.output_file):
            old_results_df = pd.read_csv(self.output_file)
            results_df = pd.concat([old_results_df, results_df])

        # Save results for run to csv
        results_df.to_csv(self.output_file, index = False)