import psycopg2
import pandas as pd
import multiprocessing as mp

import config as conf
import functions.xml_parsing_functions as xml_funcs

if __name__ == "__main__":

    print()

    # Read file list into pandas dataframe
    print('Reading file list into pandas df')
    file_list_df = pd.read_csv(conf.MASTER_FILE_LIST)
    print(file_list_df.head())

    # Extract article paths and PMC IDs as lists
    print('Extracting PMC ID and file path')
    article_paths = file_list_df['Article File'].to_list()
    pmc_ids = file_list_df['AccessionID'].to_list()

    # Connect to postgresql server
    con = psycopg2.connect(f'dbname={conf.DB_NAME} user={conf.USER} password={conf.PASSWD} host={conf.HOST}')
    cur = con.cursor()

    # Next we need a table for the filepaths of all the data files. First check to see if it exists already by 
    # trying to select it
    cur.execute("select exists(select * from information_schema.tables where table_name=%s)", ('article_file_paths',))
    data_exists = cur.fetchone()[0]

    # If We get nothing back, we need to create the table
    if data_exists == False:
        print('Creating SQL table with PMC ID and file path')

        # Create file path table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS article_file_paths(
            pmc_id varchar(12), article_file_path varchar(29))
        """)
        
        # Add file path and PMC ID
        cur.executemany("""
            INSERT INTO article_file_paths(
            pmc_id, article_file_path) VALUES(%s, %s)""", 
            zip(pmc_ids, article_paths)
        )

        con.commit()

    # If the file path table already exists, do nothing
    elif data_exists == True:
        print('File path table exists')
    
    # Now check and see if we have already parsed any articles by looking at the titles table
    cur.execute("select exists(select * from information_schema.tables where table_name=%s)", ('titles',))
    data_exists = cur.fetchone()[0]

    # If the titles table exists, we need to join it with the file path table so that we can get
    # a list of the paths to articles we have not parsed yet, i.e. PMC IDs which
    # are present in the file path table, but not in the titles table.
    if data_exists == True:
        print('We have already parsed some articles')

        # Get only file paths we have not parsed yet
        cur.execute("""
            SELECT article_file_path
            FROM article_file_paths
            LEFT OUTER JOIN titles
            ON (article_file_paths.pmc_id = titles.pmc_id)
            WHERE titles.pmc_id IS NULL
        """)

        # Get the result
        data_paths_result = cur.fetchall()

        # Take path entry only and output as list
        data_paths = [path[0] for path in data_paths_result]


    # If we have not parsed any articles yet, use the full list of article paths
    elif data_exists == False:
        print('No title data exists yet')
        data_paths = article_paths

    # Create tables for results
    print('Creating tables for output')

    cur.execute("""
        CREATE TABLE IF NOT EXISTS subjects(
        pmc_id varchar(12), subject text)
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS titles(
        pmc_id varchar(12), title text)
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS abstracts(
        pmc_id varchar(12), abstract text)
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS refs(
        pmc_id varchar(12), ref text)
    """)

    con.commit()
    con.close()

    # Prepend rest of the path to each entry in the path list
    data_paths = [f'{conf.DATA_DIR}/{i}' for i in data_paths]

    print(f'Article path list contains {len(data_paths)} files')

    # If there are article still be parsed, divide them up and submit to 
    # parse function via multiprocessing
    if len(data_paths) > 0:
        print(f'Example path: {data_paths[0]}')

        # Empty list to contain workunits
        workunits = []

        # Loop on paths, splitting them into chunks of workunit_size
        while data_paths:
            workunit, data_paths = data_paths[:conf.WORKUNIT_SIZE], data_paths[conf.WORKUNIT_SIZE:]
            workunits.append(workunit)

        print(f'Workunits created, have {len(workunits)} units of {len(workunits[0])} articles each')
        print()

        # If we have fewer workunits to parse than workers, then set the number of workers to
        # the number of workunits so that each worker gets one workunit and then we are done.
        if len(workunits) < conf.NUM_WORKERS:
            workers = len(workunits)

        print(f'Spawning {conf.NUM_WORKERS} workers')

        # Initialize pool and submit 
        pool = mp.Pool(conf.NUM_WORKERS)
        results = pool.map(xml_funcs.parse_pmc_xml, workunits)

        # Close up shop
        pool.close()
        pool.join()
        print()
    
    else:
        print('Nothing to do, done')
        print()