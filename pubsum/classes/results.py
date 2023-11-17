import os.path
import pandas as pd
import config as conf

class Results:
    '''Class to hold objects related to summarization
    benchmarking results'''

    def __init__(self):

        # Output file for results
        self.output_file = f'{conf.summarization_benchmark_results_dir}/{conf.summarization_benchmark_output_file_name}'

        # Independent vars for run
        self.data = {}
        self.data['abstract_num'] = []
        self.data['summarization_time'] = []
        self.data['device_map_strategy'] = []

    def save_result(self):

        # Make dataframe of new results
        results_df = pd.DataFrame(self.data)

        # Read existing results if any and concatenate new results
        if os.path.exists(self.output_file):
            old_results_df = pd.read_csv(self.output_file)
            results_df = pd.concat([old_results_df, results_df])

        # Save results for run to csv
        results_df.to_csv(self.output_file, index = False)
