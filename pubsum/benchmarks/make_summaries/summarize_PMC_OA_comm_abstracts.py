import time
import psycopg2
import psycopg2.extras
import torch
import datetime
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig

import config as conf

if __name__ == "__main__":

    # Connect to postgresql server
    con = psycopg2.connect(f'dbname={conf.DB_NAME} user={conf.USER} password={conf.PASSWD} host={conf.HOST}')
    cur = con.cursor()

    # Next we need a table for the abstract summaries. First check to see if it exists already by 
    # trying to select it
    cur.execute("select exists(select * from information_schema.tables where table_name=%s)", ('abstract_summaries',))
    data_exists = cur.fetchone()[0]

    # If we get nothing back, we need to create the table
    if data_exists == False:
        print('\nCreating SQL table with PMC ID and abstract summary\n')

        # Create file path table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS abstract_summaries(
            pmc_id varchar(12), abstract_summary text)
        """)

        con.commit()

    # If the file path table already exists, do nothing
    else:
        print('\nAbstract summary table exists\n')

    # Then open another connection and create a cursor for writing
    con2 = psycopg2.connect(f'dbname={conf.DB_NAME} user={conf.USER} password={conf.PASSWD} host={conf.HOST}')

    # Initialize model set-up generation config
    model = AutoModelForSeq2SeqLM.from_pretrained("haining/scientific_abstract_simplification", device_map = 'auto')
    
    gen_cfg = GenerationConfig.from_model_config(model.config)
    gen_cfg.max_length = 256
    gen_cfg.top_p = 0.9
    gen_cfg.do_sample = True
    gen_cfg.torch_dtype = torch.bfloat16

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("haining/scientific_abstract_simplification")

    # Prompt to prepend to abstracts
    INSTRUCTION = "summarize, simplify, and contextualize: "

    # Do reps for science
    reps = 3
    total_rows = 24
    chunk_sizes = [24, 12, 6, 3]

    results = {
        'total_times': [],
        'summarization_times': [],
        'insert_times': [],
        'loading_times': [],
        'chunk_size': [],
        'replicate': []
    }

    for chunk_size in chunk_sizes:

        for i in range(reps):

            # Create a server side cursor to iterate through the abstracts
            reader = con.cursor('reader', cursor_factory = psycopg2.extras.DictCursor)
            reader.itersize = chunk_size
            reader.execute('SELECT * FROM abstracts LIMIT %s', (total_rows,))

            # Create a second cursor for writing
            writer = con2.cursor()

            # Loop on abstracts and summarize
            row_count = 0
            loop_start = time.time()
            summarization_time = 0
            insert_time = 0

            for row in reader:

                if row[1] != None:

                    row_count += 1

                    print(f'Chunk size: {chunk_size}, rep {i}, row: {row_count}', end='\r')

                    summarization_start = time.time()

                    encoding = tokenizer(
                        INSTRUCTION + row[1], 
                        max_length = 672, 
                        padding = 'max_length', 
                        truncation = True, 
                        return_tensors = 'pt'
                    ).to('cuda')
                    
                    decoded_ids = model.generate(
                        input_ids = encoding['input_ids'],
                        attention_mask = encoding['attention_mask'], 
                        generation_config = gen_cfg
                    )
                    
                    summary = tokenizer.decode(decoded_ids[0], skip_special_tokens = True)
                    
                    summarization_end = time.time()
                    summarization_time += summarization_end - summarization_start

                    insert_start = time.time()

                    writer.execute("INSERT INTO abstract_summaries (pmc_id, abstract_summary) VALUES(%s, %s)", (row[0], summary))
                    con2.commit()

                    insert_end = time.time()
                    insert_time += insert_end - insert_start

            # Close our cursors for next loop
            reader.close()
            writer.close()

            loop_end = time.time()

            total_time = (loop_end - loop_start) / row_count
            summarization_time = summarization_time / row_count
            insert_time = insert_time / row_count
            loading_time = total_time - summarization_time - insert_time

            results['replicate'].append(i)
            results['chunk_size'].append(chunk_size)
            results['total_times'].append(total_time)
            results['summarization_times'].append(summarization_time)
            results['insert_times'].append(insert_time)
            results['loading_times'].append(loading_time)

            print(f'\nReplicate: {i}')
            print(f'Chunk size: {chunk_size}')
            print(f'Total time: {total_time}')
            print(f'Summarization time: {summarization_time}')
            print(f'Insert time: {insert_time}')
            print(f'Loading time: {loading_time}\n')

    results_df = pd.DataFrame(results)
    print(results_df)
    print()
    print(results_df.info())

    result_date = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M')
    result_file = f'./testing/SQL_benchmark_results/{result_date}_GPU.csv'
    results_df.to_csv(result_file, index=False)