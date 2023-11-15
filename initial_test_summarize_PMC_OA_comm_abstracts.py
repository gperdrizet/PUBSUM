import time
import config as conf
from classes.database import Database
from classes.llm import Llm
from classes.results import Results

if __name__ == "__main__":

    # Get SQL database connections ready
    database = Database()

    # Fire up the model
    llm = Llm()

    # Instantiate collector for results
    results = Results()

    for use_gpu in conf.use_gpu:
        for num_jobs in conf.num_jobs:

            # Loop on abstract rows
            row_count = 0
            for row in database.cur2:

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

        # Outermost independent variable loop is done, so save the results for
        # this condition
        results.save_result()