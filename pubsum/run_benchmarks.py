import argparse
import os
import config as conf
import benchmarks.load_summarize_insert.benchmark as lsi
import benchmarks.sql_insert.benchmark as sql
import benchmarks.huggingface_device_map.benchmark as device_map
import benchmarks.model_quantization.benchmark as quantization
import benchmarks.parallel_summarize.benchmark as parallel
import benchmarks.batched_inference.benchmark as batched_inference
import benchmarks.parallel_batched_summarize.benchmark as parallel_batched

###########################
# Benchmarking parameters #
###########################

# Benchmarks parent dir
benchmark_dir = f'{conf.PROJECT_ROOT_PATH}/benchmarks/'

# Initial MVP load, summarize, insert execution time benchmark
summarize_benchmark_results_dir = f'{benchmark_dir}/load_summarize_insert'
summarize_benchmark_abstracts = 5
summarize_benchmark_replicates = 30

# PostgreSQL/psycopg2 insert benchmark for table creation
insert_benchmark_results_dir = f'{benchmark_dir}/sql_insert'
insert_benchmark_abstracts = [10000, 200000, 400000]
insert_benchmark_replicates = 10
insert_strategies = ['execute_many', 'execute_batch', 'execute_values', 'mogrify', 'stringIO']

# Huggingface device map benchmark for abstract summarization
device_map_benchmark_results_dir = f'{benchmark_dir}/huggingface_device_map'
device_map_benchmark_abstracts = 16
device_map_strategies = ['CPU only', 'multi-GPU', 'single GPU', 'balanced', 'balanced_low_0', 'sequential']

# Huggingface model quantization benchmark
model_quantization_benchmark_results_dir = f'{benchmark_dir}/model_quantization'
model_quantization_benchmark_abstracts = 10
model_quantization_benchmark_quantization_strategies = [
    'none',
    'eight bit', 
    'four bit', 
    'four bit nf4',
    'nested four bit',
    'nested four bit nf4',
    'none + BT',
    'eight bit + BT', 
    'four bit + BT', 
    'four bit nf4 + BT',
    'nested four bit + BT',
    'nested four bit nf4 + BT',
]

# Batched inference benchmark
batched_inference_benchmark_results_dir = f'{benchmark_dir}/batched_inference'
batched_inference_benchmark_abstracts = 16
batched_inference_benchmark_replicates = 5
batched_inference_benchmark_batch_sizes = [1, 2, 4, 8, 16]

# Data parallel summarization benchmark
parallel_summarize_benchmark_results_dir = f'{benchmark_dir}/parallel_summarize'
parallel_summarize_benchmark_abstracts = 120
parallel_summarize_benchmark_replicates = 5
parallel_summarize_device_map_strategies = ['GPU', 'CPU physical cores', 'CPU hyperthreading']
parallel_summarize_num_CPU_jobs = [1, 2, 5, 10, 20]
parallel_summarize_num_GPU_jobs = [4, 8, 12] # More than 3 jobs on a single GK210 crashes OOM.
parallel_summarize_gpus = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'] # Available GPUs

# Data parallel batched summarization benchmark
parallel_batched_summarize_benchmark_results_dir = f'{benchmark_dir}/parallel_batched_summarize'
parallel_batched_summarize_benchmark_rounds = 3
parallel_batched_summarize_benchmark_replicates = 5
parallel_batched_summarize_benchmark_batch_size = [1, 2, 4, 8, 16]
parallel_batched_summarize_num_GPU_jobs = [4, 8, 12, 16]
parallel_batched_summarize_gpus = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'] # Available GPUs

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
        '--model_quantization',
        choices=[str(True), str(False)],
        default=str(False),
        help='Run model quantization benchmark?'
    )

    parser.add_argument(
        '--batched_inference',
        choices=[str(True), str(False)],
        default=str(False),
        help='Run inference batch size benchmark?'
    )

    parser.add_argument(
        '--parallel_summarize',
        choices=[str(True), str(False)],
        default=str(False),
        help='Run data parallel summarization benchmark?'
    )

    parser.add_argument(
        '--parallel_batched_summarize',
        choices=[str(True), str(False)],
        default=str(False),
        help='Run data parallel batched summarization benchmark?'
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

    # Model quantization benchmark
    if args.model_quantization == 'True':

        # Silence parallelism warning
        os.environ['TOKENIZERS_PARALLELISM'] = 'true'

        quantization.benchmark(
            conf.DB_NAME,
            conf.USER,
            conf.PASSWD, 
            conf.HOST,
            args.resume,
            model_quantization_benchmark_results_dir,
            model_quantization_benchmark_abstracts,
            model_quantization_benchmark_quantization_strategies
        )

    # Batched inference benchmark
    if args.batched_inference == 'True':

        batched_inference.benchmark(
            conf.DB_NAME,
            conf.USER,
            conf.PASSWD, 
            conf.HOST,
            args.resume,
            batched_inference_benchmark_results_dir,
            batched_inference_benchmark_abstracts,
            batched_inference_benchmark_replicates,
            batched_inference_benchmark_batch_sizes
        )


    # Data parallel summarization benchmark
    if args.parallel_summarize == 'True':

        parallel.benchmark(
            conf.DB_NAME,
            conf.USER,
            conf.PASSWD, 
            conf.HOST,
            args.resume,
            parallel_summarize_benchmark_results_dir,
            parallel_summarize_benchmark_abstracts,
            parallel_summarize_benchmark_replicates,
            parallel_summarize_device_map_strategies,
            parallel_summarize_num_CPU_jobs,
            parallel_summarize_num_GPU_jobs,
            parallel_summarize_gpus,
        )

    # Data parallel summarization benchmark
    if args.batched_parallel_summarize == 'True':

        parallel_batched.benchmark(
            conf.DB_NAME,
            conf.USER,
            conf.PASSWD, 
            conf.HOST,
            args.resume,
            parallel_batched_summarize_benchmark_results_dir,
            parallel_batched_summarize_benchmark_rounds,
            parallel_batched_summarize_benchmark_replicates,
            parallel_batched_summarize_benchmark_batch_size,
            parallel_batched_summarize_num_GPU_jobs,
            parallel_batched_summarize_gpus
        )