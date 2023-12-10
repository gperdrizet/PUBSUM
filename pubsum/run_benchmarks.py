import argparse
import config as conf
import benchmarks.helper_functions as helper_funcs
import benchmarks.baseline_execute_time.benchmark as baseline_execute
import benchmarks.sql_insert.benchmark as sql
import benchmarks.huggingface_device_map.benchmark as device_map
import benchmarks.model_quantization.benchmark as quantization
import benchmarks.parallel_summarization.benchmark as parallel
import benchmarks.batched_summarization.benchmark as batched_summarization
import benchmarks.parallel_batched_summarization.benchmark as parallel_batched

if __name__ == "__main__":

    ############################################################
    # Command line args used to select which benchmarks to run #
    ############################################################

    arguments = [
        ['--baseline_execute', 'Run MVP load, summarize, insert benchmark?'],
        ['--sql_insert', 'Run sql insert benchmark?'],
        ['--hf_device_map', 'Run huggingface device map benchmark?'],
        ['--model_quantization', 'Run model quantization benchmark?'],
        ['--batched_summarization', 'Run batched summarization benchmark?'],
        ['--parallel_summarization', 'Run data parallel summarization benchmark?'],
        ['--parallel_batched_summarization', 'Run data parallel batched summarization benchmark?'],
        ['--run_all', 'Run all benchmarks?'],
        ['--resume', 'Resume prior run and append data?']
    ]

    # Instantiate command line argument parser
    parser = argparse.ArgumentParser(
        prog = 'run_benchmarks.py',
        description = 'Launcher for project benchmarks. Choose which to run via command line args:',
        epilog = 'Bottom text'
    )
    
    # Add arguments
    for argument in arguments:

        parser.add_argument(
            argument[0], 
            choices=[str(True), str(False)], 
            default=str(False), 
            help=argument[1]
        )

    # Parse the arguments
    args = parser.parse_args()

    #############################################
    # Run the benchmarks called for by the user #
    #############################################

    # Initial load, summarize, insert timing benchmark
    if args.baseline_execute == 'True' or args.run_all == 'True':

        baseline_execute.benchmark(
            helper_funcs=helper_funcs,
            resume=args.resume,
            results_dir=f'{conf.BENCHMARK_DIR}/baseline_execute_time',
            num_abstracts=5,
            replicates=5,
            **conf.SQL_SERVER_KWARGS
        )

    # SQL insert benchmark
    if args.sql_insert == 'True' or args.run_all == 'True':

        sql.benchmark(
            helper_funcs=helper_funcs,
            resume=args.resume,
            master_file_list=conf.MASTER_FILE_LIST,
            results_dir=f'{conf.BENCHMARK_DIR}/sql_insert',
            abstract_nums=[10000, 200000, 400000],
            replicates=5,
            insert_strategies=[
                'execute_many', 
                'execute_batch', 
                'execute_values', 
                'mogrify', 
                'stringIO'
            ],
            **conf.SQL_SERVER_KWARGS
        )

    # Huggingface device map strategy for summarization benchmark
    if args.hf_device_map == 'True' or args.run_all == 'True':

        device_map.benchmark(
            helper_funcs=helper_funcs,
            resume=args.resume,
            results_dir=f'{conf.BENCHMARK_DIR}/huggingface_device_map',
            replicates=5,
            num_abstracts=3,
            device_map_strategies=[
                'CPU only', 
                'multi-GPU', 
                'single GPU', 
                'balanced', 
                'balanced_low_0', 
                'sequential'
            ],
            **conf.SQL_SERVER_KWARGS
        )

    # Model quantization benchmark
    if args.model_quantization == 'True' or args.run_all == 'True':

        quantization.benchmark(
            helper_funcs=helper_funcs,
            resume=args.resume,
            results_dir=f'{conf.BENCHMARK_DIR}/model_quantization',
            replicates=5,
            num_abstracts=3,
            quantization_strategies=[
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
            ],
            **conf.SQL_SERVER_KWARGS
        )

    # Batched summarization benchmark
    if args.batched_summarization == 'True' or args.run_all == 'True':

        batched_summarization.benchmark(
            helper_funcs=helper_funcs,
            resume=args.resume,
            results_dir=f'{conf.BENCHMARK_DIR}/batched_summarization',
            replicates=5,
            batches=3,
            batch_sizes=[1, 2, 4, 8, 16, 32, 64],
            quantization_strategies=['none', 'four bit'],
            **conf.SQL_SERVER_KWARGS
        )

    # Data parallel summarization benchmark
    if args.parallel_summarization == 'True' or args.run_all == 'True':

        parallel.benchmark(
            resume=args.resume,
            results_dir=f'{conf.BENCHMARK_DIR}/parallel_summarization',
            replicates=5,
            batches=3,
            devices=['GPU', 'CPU'],
            workers=[1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
            gpus=conf.GPUS,
            **conf.SQL_SERVER_KWARGS
        )

    # Data parallel batched summarization benchmark
    if args.parallel_batched_summarization == 'True' or args.run_all == 'True':

        parallel_batched.benchmark(
            resume=args.resume,
            results_dir=f'{conf.BENCHMARK_DIR}/parallel_batched_summarization',
            replicates=5,
            batches=3,
            batch_sizes=[16, 32, 64],
            workers=[4, 8, 12, 16, 20, 24],
            gpus=conf.GPUS,
            quantization_strategies=['none', 'four bit'],
            **conf.SQL_SERVER_KWARGS
        )