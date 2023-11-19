import argparse
import config as conf
import benchmarks.load_summarize_insert.benchmark as lsi
import benchmarks.sql_insert.benchmark as sql
import benchmarks.huggingface_device_map.benchmark as device_map

###########################
# Benchmarking parameters #
###########################

# Benchmarks parent dir
benchmark_dir = f'{conf.PROJECT_ROOT_PATH}/benchmarks/'

# Initial MVP load, summarize, insert execution time benchmark
summarize_benchmark_results_dir = f'{benchmark_dir}/load_summarize_insert'
summarize_benchmark_abstracts = 3
summarize_benchmark_replicates = 3

# PostgreSQL/psycopg2 insert benchmark for table creation
insert_benchmark_results_dir = f'{benchmark_dir}/sql_insert'
insert_benchmark_abstracts = [1000, 20000, 40000]
insert_strategies = ['execute_many', 'execute_batch', 'execute_values', 'mogrify', 'stringIO']
insert_benchmark_replicates = 3

# Huggingface device map benchmark for abstract summarization
device_map_benchmark_results_dir = f'{benchmark_dir}/huggingface_device_map'
device_map_benchmark_abstracts = 3
device_map_strategies = ['CPU only', 'multi-GPU', 'single GPU', 'balanced', 'balanced_low_0', 'sequential']

if __name__ == "__main__":

    ############################################################
    # Command line args used to select which benchmarks to run #
    ############################################################

    # Instantiate command line argument parser
    parser = argparse.ArgumentParser(
        prog = 'run_benchmarks.py',
        description = 'Launcher for project benchmarks. Choose which to run via command line args:',
        epilog = 'Bottom text'
    )
    
    # Add arguments
    parser.add_argument(
        '--load_summarize_insert', 
        choices=[str(True), str(False)], 
        default=str(False), 
        help='Run MVP load, summarize, insert benchmark?'
    )

    parser.add_argument(
        '--sql_insert', 
        choices=[str(True), str(False)], 
        default=str(False), 
        help='Run sql insert benchmark?'
    )

    parser.add_argument(
        '--hf_device_map',
        choices=[str(True), str(False)],
        default=str(False),
        help='Run huggingface device map benchmark?'
    )

    parser.add_argument(
        '--resume', 
        choices=[str(True), str(False)], 
        default=str(False), 
        help='Resume prior run and append data?'
    )

    # Parse the arguments
    args = parser.parse_args()

    #############################################
    # Run the benchmarks called for by the user #
    #############################################

    # Initial load, summarize, insert timing benchmark
    if args.load_summarize_insert == 'True':

        lsi.benchmark(
            conf.DB_NAME,
            conf.USER,
            conf.PASSWD, 
            conf.HOST,
            args.resume,
            summarize_benchmark_results_dir,
            summarize_benchmark_abstracts,
            summarize_benchmark_replicates
        )

    # SQL insert benchmark
    if args.sql_insert == 'True':

        sql.benchmark(
            conf.MASTER_FILE_LIST,
            conf.DB_NAME,
            conf.USER,
            conf.PASSWD, 
            conf.HOST,
            args.resume,
            insert_benchmark_results_dir,
            insert_benchmark_abstracts,
            insert_strategies,
            insert_benchmark_replicates
        )

    # Huggingface device map strategy for summarization benchmark
    if args.hf_device_map == 'True':

        device_map.benchmark(
            conf.DB_NAME,
            conf.USER,
            conf.PASSWD, 
            conf.HOST,
            args.resume,
            device_map_benchmark_results_dir,
            device_map_benchmark_abstracts,
            device_map_strategies
        )