import time

# Janky, yes, but it works - this script is a throwaway and won't exist
# if this project ever becomes a package
import os
import sys
sys.path.append(f'{os.path.dirname(os.path.realpath(__file__))}/..')

import config as conf
from classes.database import Database
from classes.llm import Llm
from classes.results import Results

if __name__ == "__main__":

    # Instantiate collector for results
    results = Results()

    # Start a write cursor and make a table for
    # the summaries
    database = Database()
    database.create_write_cursor()
    database.make_summary_table()

    for use_gpu in conf.use_gpu:
        for num_jobs in conf.num_jobs:

            print(f'\nStarting benchmark run with use_gpu = {use_gpu} and {num_jobs} jobs.')

            # Fire up the model for this run
            llm = Llm(use_gpu)
            
            # Start new reader cursor
            database.create_read_cursor()

            # Get the rows
            database.get_rows()

            # Loop on abstract rows
            row_count = 0

            for row in database.read_cur:

                # Make sure this abstract actually has content to be summarized
                if row[1] != None:

                    # Plink
                    row_count += 1
                    results.data['abstract_num'].append(row_count)
                    results.data['used_gpu'].append(use_gpu)
                    results.data['num_jobs'].append(num_jobs)
                    pmcid = row[0]
                    print(f'Summarizing {pmcid}: {row_count} of {conf.num_abstracts}', end = '\r')

                    # Do and time the summary
                    summarization_start = time.time()
                    summary = llm.summarize(row[1])
                    results.data['summarization_time'].append(time.time() - summarization_start)

                    # Insert the new summary into the table
                    database.insert(summary, pmcid)

        # Innermost independent variable loop is done, so save the results for
        # this condition and close the read cursor
        results.save_result()
        database.close_read_cursor()