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
import benchmarks.optimized_execute_time as optimized_execute
from multiprocessing import Process, Queue

if __name__ == "__main__":

    ############################################################
    # Command line args used to select which benchmarks to run #
    ############################################################

    arguments = [
        ['--baseline_execute', 'Run MVP load, summarize, insert benchmark?'],
        ['--hf_device_map', 'Run huggingface device map benchmark?'],
        ['--parallel_summarization', 'Run data parallel summarization benchmark?'],
        ['--model_quantization', 'Run model quantization benchmark?'],
        ['--batched_summarization', 'Run batched summarization benchmark?'],
        ['--parallel_batched_summarization', 'Run data parallel batched summarization benchmark?'],
        ['--sql_insert', 'Run sql insert benchmark?'],
        #['--optimized_execute', 'Run optimized load, summarize, insert benchmark?'],
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

    # Create multiprocessing queue so we can run each benchmark
    # in it's own subprocess. This makes sure that any artifacts
    # e.g. CUDA context die with the benchmark and don't
    # interfere with subsequent tests
    queue = Queue()

    # Initial load, summarize, insert timing benchmark
    if args.baseline_execute == 'True' or args.run_all == 'True':

        p = Process(target=baseline_execute.benchmark,
            kwargs=dict(
                helper_funcs=helper_funcs,
                resume=args.resume,
                results_dir=f'{conf.BENCHMARK_DIR}/baseline_execute_time',
                replicates=300,
                db_name=conf.DB_NAME,
                user=conf.USER,
                passwd=conf.PASSWD,
                host=conf.HOST
            )
        )

        p.start()
        p.join()

    # Huggingface device map strategy for summarization benchmark
    if args.hf_device_map == 'True' or args.run_all == 'True':

        p = Process(target=device_map.benchmark,
            kwargs=dict(
                helper_funcs=helper_funcs,
                resume=args.resume,
                results_dir=f'{conf.BENCHMARK_DIR}/huggingface_device_map',
                replicates=10,
                device_map_strategies=[
                    'CPU only',
                    'multi-GPU',
                    'single GPU',
                    'balanced',
                    #'balanced_low_0',
                    'sequential'
                ],
                db_name=conf.DB_NAME,
                user=conf.USER,
                passwd=conf.PASSWD,
                host=conf.HOST
            )
        )

        p.start()
        p.join()

    # Data parallel summarization benchmark
    if args.parallel_summarization == 'True' or args.run_all == 'True':

        p = Process(target=parallel.benchmark,
            kwargs=dict(
                resume=args.resume,
                results_dir=f'{conf.BENCHMARK_DIR}/parallel_summarization',
                replicates=5,
                devices=[
                    'GPU',
                    'GPU: sequential',
                    'CPU: 1 thread per worker',
                    'CPU: 2 threads per worker',
                    'CPU: 4 threads per worker'
                ],
                workers=[1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20],
                gpus=conf.GPUS,
                db_name=conf.DB_NAME,
                user=conf.USER,
                passwd=conf.PASSWD,
                host=conf.HOST
            )
        )

        p.start()
        p.join()

    # Model quantization benchmark
    if args.model_quantization == 'True' or args.run_all == 'True':

        p = Process(target=quantization.benchmark,
                kwargs=dict(
                helper_funcs=helper_funcs,
                resume=args.resume,
                results_dir=f'{conf.BENCHMARK_DIR}/model_quantization',
                replicates=10,
                num_abstracts=3,
                quantization_strategies=[
                    'none',
                    'eight bit',
                    'four bit',
                    'four bit nf4','max memory allocated (GB)'
                    'nested four bit',
                    'nested four bit nf4',
                    'none + BT',
                    'eight bit + BT',
                    'four bit + BT',
                    'four bit nf4 + BT',
                    'nested four bit + BT',
                    'nested four bit nf4 + BT',
                ],
                db_name=conf.DB_NAME,
                user=conf.USER,
                passwd=conf.PASSWD,
                host=conf.HOST
            )
        )

        p.start()
        p.join()

    # Batched summarization benchmark
    if args.batched_summarization == 'True' or args.run_all == 'True':

        p = Process(target=batched_summarization.benchmark,
            kwargs=dict(
                helper_funcs=helper_funcs,
                resume=args.resume,
                results_dir=f'{conf.BENCHMARK_DIR}/batched_summarization',
                replicates=3,
                batches=3,
                batch_size_lists=[[1, 2, 4, 8, 16],[27, 28, 29, 30], [32, 64, 81, 82, 83]],
                quantization_strategy_lists=[['none', 'four bit nf4'], ['none'], ['four bit nf4']],
                db_name=conf.DB_NAME,
                user=conf.USER,
                passwd=conf.PASSWD,
                host=conf.HOST
            )
        )

        p.start()
        p.join()

    # Data parallel batched summarization benchmark
    if args.parallel_batched_summarization == 'True' or args.run_all == 'True':

        p = Process(target=parallel_batched.benchmark,
            kwargs=dict(
                resume=args.resume,
                results_dir=f'{conf.BENCHMARK_DIR}/parallel_batched_summarization',
                replicates=3,
                batches=3,
                batch_size_lists=[
                    [1, 2, 4, 8, 16, 28, 29, 30],         # none, 1
                    [1, 2, 4, 7, 8, 9],                   # none, 2 
                    [1, 2],                               # none, 3
                    [1],                                  # none, 4
                    [1, 2, 4, 8, 16, 32, 64, 83, 84, 85], # four bit, 1
                    [1, 2, 4, 8, 16, 33, 34, 35],         # four bit, 2
                    [1, 2, 4, 8, 16, 18, 19, 20],         # four bit, 3
                    [1, 2, 4, 8, 10, 11, 12],             # four bit, 4  
                    [1, 2, 4, 5, 6, 7],                   # four bit, 5
                    [1, 2, 3, 4],                         # four bit, 6
                    [1, 2, 3],                            # four bit, 7
                    [1]                                   # four bit, 8 
                ],
                worker_count_lists=[
                    [1],
                    [2],
                    [3],
                    [4],
                    [1],
                    [2],
                    [3],
                    [4],
                    [5],
                    [6],
                    [7],
                    [8]
                ],
                gpus=conf.GPUS,
                quantization_strategy_lists=[
                    ['none'],
                    ['none'],
                    ['none'],
                    ['none'],
                    ['four bit'],
                    ['four bit'],
                    ['four bit'],
                    ['four bit'],
                    ['four bit'],
                    ['four bit'],
                    ['four bit'],
                    ['four bit']
                ],
                db_name=conf.DB_NAME,
                user=conf.USER,
                passwd=conf.PASSWD,
                host=conf.HOST,
                file_name='single_GPU_results.csv'
            )
        )

        p.start()
        p.join()

    # SQL insert benchmark
    if args.sql_insert == 'True' or args.run_all == 'True':

        p = Process(target=sql.benchmark,
            kwargs=dict(
                helper_funcs=helper_funcs,
                resume=args.resume,
                master_file_list=conf.MASTER_FILE_LIST,
                results_dir=f'{conf.BENCHMARK_DIR}/sql_insert',
                replicates=50,
                abstract_nums=[10000, 200000, 400000],
                insert_strategies=[
                    'execute_many',
                    'execute_batch',
                    'execute_values',
                    'mogrify',
                    'stringIO'
                ],
                db_name=conf.DB_NAME,
                user=conf.USER,
                passwd=conf.PASSWD,
                host=conf.HOST
            )
        )

        p.start()
        p.join()

    # # Optimized load, summarize, insert timing benchmark
    # if args.optimized_execute == 'True' or args.run_all == 'True':

    #     p = Process(target=optimized_execute.benchmark,
    #         kwargs=dict(
    #             helper_funcs=helper_funcs,
    #             resume=args.resume,
    #             results_dir=f'{conf.BENCHMARK_DIR}/baseline_execute_time',
    #             replicates=50,
    #             num_abstracts=3,
    #             db_name=conf.DB_NAME,
    #             user=conf.USER,
    #             passwd=conf.PASSWD,
    #             host=conf.HOST
    #         )
    #     )

    #     p.start()
    #     p.join()
