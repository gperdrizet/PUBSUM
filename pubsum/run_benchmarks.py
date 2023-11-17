import argparse
import config as conf
import benchmarks.sql_insert.benchmark as sql
import benchmarks.huggingface_device_map.benchmark as device_map

if __name__ == "__main__":

    # Instantiate command line argument parser
    parser = argparse.ArgumentParser(
        prog = 'run_benchmarks.py',
        description = 'Launcher for project benchmarks. Choose which to run via command line args:',
        epilog = 'Bottom text')
    
    # Add arguments
    parser.add_argument('--sql_insert', choices=[str(True), str(False)], default=False, help='BOOL, run sql insert benchmark?')
    parser.add_argument('--hf_device_map', choices=[str(True), str(False)], default=False, help='BOOL, run huggingface device map benchmark?')

    # Parse the arguments
    args = parser.parse_args()

    #############################################
    # Run the benchmarks called for by the user #
    #############################################

    # SQL insert benchmark
    if bool(args.sql_insert) == True:

        sql.benchmark(
            conf.MASTER_FILE_LIST,
            conf.DB_NAME,
            conf.USER,
            conf.PASSWD, 
            conf.HOST
        )

    # Huggingface device map strategy for summarization benchmark
    if bool(args.hf_device_map) == True:

        device_map.benchmark(
            conf.DB_NAME,
            conf.USER,
            conf.PASSWD, 
            conf.HOST,
            conf.device_map_strategies,
            conf.device_map_benchmark_abstracts,
            conf.device_map_benchmark_results_dir
        )