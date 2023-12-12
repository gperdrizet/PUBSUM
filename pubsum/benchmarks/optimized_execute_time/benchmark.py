import time
import os
import gc
import psycopg2
import psycopg2.extras
import pandas as pd
import torch
from typing import List, Tuple
import transformers

def benchmark(
    helper_funcs,
    resume: bool,
    results_dir: str, 
    num_abstracts: int, 
    replicates: int,
    db_name: str, 
    user: str, 
    passwd: str, 
    host: str
):

    print(f'\nRunning baseline execution time benchmark. Resume = {resume}.')

    # Keys for data collection
    collection_vars = [
        'replicate',
        'num_abstracts',
        'total_replicate_time',
        'total_summarization_time',
        'total_insert_time',
        'total_loading_time',
        'mean_total_time',
        'mean_summarization_time',
        'mean_insert_time',
        'mean_loading_time' 
    ]

    # Pick up where we left off if asked and old data exists
    replicate = resume_run(
        helper_funcs,
        resume=resume,
        results_dir=results_dir,
        collection_vars=collection_vars,
        replicates=replicates
    )

    if replicate < replicates:

        # Connect to postgresql server
        print('Connecting to SQL server.')
        connection = psycopg2.connect(f'dbname={db_name} user={user} password={passwd} host={host}')

        # Make table for results
        make_target_table(connection)

        # Fire up the LLM
        model, tokenizer, gen_cfg = start_llm()

        # Loop on replicates
        while replicate <= replicates:

            print(f'\nBaseline execution time benchmark:\n')
            print(f' Replicate: {replicate} of {replicates}')
            print(f' Abstracts: {num_abstracts}\n')

            # Create result collector for this replicate
            results = helper_funcs.Results(
                results_dir=results_dir,
                collection_vars=collection_vars
            )

            # Create a cursor for writing
            write_cursor = connection.cursor()

            # Loop on abstracts in this replicate and time everything
            row_count = 0
            replicate_start = time.time()
            total_summarization_time = 0
            total_insert_time = 0

            # Get rows from abstracts table
            rows = helper_funcs.get_rows(
                db_name=db_name, 
                user=user, 
                passwd=passwd, 
                host=host, 
                num_abstracts=num_abstracts
            )

            for row in rows:

                row_count += 1

                print(f' Processing abstract {row_count} of {num_abstracts}.')

                # Parse row into article accession number and abstract text
                pmcid = row[0]
                abstract = row[1]

                # Time the LLM doing it's thing
                summarization_start = time.time()

                # Do the summary
                summary = summarize(
                    abstract=abstract,
                    model=model, 
                    tokenizer=tokenizer, 
                    gen_cfg=gen_cfg
                )
                    
                # Stop the summarization timer and add the dT to the total
                summarization_end = time.time()
                total_summarization_time += summarization_end - summarization_start

                # Start the SQL insert timer
                insert_start = time.time()

                # Insert the summary text into the SQL table
                write_cursor.execute(
                    "INSERT INTO summary_benchmark (pmc_id, abstract_summary) VALUES(%s, %s)", 
                    (pmcid, summary)
                )

                connection.commit()

                # Stop the insert timer and add the dT to the total
                insert_end = time.time()
                total_insert_time += insert_end - insert_start

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

            # Close our cursor for next replicate
            write_cursor.close()

            # Save the results from this replicate
            results.save_result()

            replicate += 1
            print(' Done.')

        # Get rid of model and tokenizer from run, free up memory
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    return True

def resume_run(
    helper_funcs,
    resume: bool,
    results_dir: str,
    collection_vars: List[str],
    replicates: int
) -> int:

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
        results = helper_funcs.Results(
            results_dir=results_dir,
            collection_vars=collection_vars
        )

        results.save_result(overwrite = True)
        replicate = 1

    return replicate

def make_target_table(connection) -> None:

    # Next, we need a table to insert results into. Make a scratch table just for this test.
    # If the table exists already, delete it.
    print('Creating target table for summaries.')
    
    write_cursor = connection.cursor()

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

def start_llm() -> Tuple[
    transformers.T5ForConditionalGeneration, 
    transformers.T5TokenizerFast, 
    transformers.GenerationConfig
]:
    
    # Initialize model and set-up generation config
    print('Initializing model and tokenizer.')

    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
        'haining/scientific_abstract_simplification'
    )
    
    gen_cfg = transformers.GenerationConfig.from_model_config(model.config)
    gen_cfg.max_length = 512
    gen_cfg.top_p = 0.9
    gen_cfg.do_sample = True

    # Initialize tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        'haining/scientific_abstract_simplification'
    )

    return model, tokenizer, gen_cfg

def summarize(
    model: transformers.T5ForConditionalGeneration,
    tokenizer: transformers.T5TokenizerFast,
    gen_cfg: transformers.GenerationConfig,
    abstract: str
) -> str:

    # Add the prompt to the abstract and tokenize
    encoding = tokenizer(
        'summarize, simplify, and contextualize: ' + abstract, 
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

    return summary