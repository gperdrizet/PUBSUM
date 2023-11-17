'''Benchmarking script to try out various methods
for inserting large numbers of rows into postgreSQL
tables via psycopg2
'''

import psycopg2
import pandas as pd

def benchmark(master_file_list, db_name, user, passwd, host):

    print('Running SQL insert benchmark.')

    # Read file list into pandas dataframe
    print('Reading file list into pandas df')
    file_list_df = pd.read_csv(master_file_list)

    # Extract article paths and PMC IDs
    print('Extracting PMC ID and file path')
    article_paths_df = pd.DataFrame(file_list_df, columns=['Article File','AccessionID'])

    # Connect to postgresql server
    connection = psycopg2.connect(f'dbname={db_name} user={user} password={passwd} host={host}')
    write_cursor = connection.cursor()

    # Next we need a table to insert into. Make a scratch table just for this test.
    # If the table exists already, delete it.
    write_cursor.execute('''
        DROP TABLE IF EXISTS insert_benchmark
    ''')

    write_cursor.commit()

    # Then create it
    write_cursor.execute('''
        CREATE TABLE IF NOT EXISTS insert_benchmark(
        pmc_id varchar(12), article_file_path varchar(29))
    ''')
    
    write_cursor.commit()