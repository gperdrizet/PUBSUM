import argparse
import config as conf
import optimization.device_map.optimization as device_map
import optimization.batch_size.optimization as batch_size
from multiprocessing import Process, Queue

if __name__ == "__main__":

    ####################################################################
    # Command line args used to select which optimization tests to run #
    ####################################################################

    arguments = [
        ['--device_map', 'Run device map optimization?'],
        ['--batch_size', 'Run batch size optimization?'],
        ['--run_all', 'Run all optimization tests?'],
        ['--resume', 'Resume prior run(s) and append data?']
    ]

    # Instantiate command line argument parser
    parser = argparse.ArgumentParser(
        prog = 'run_optimization.py',
        description = 'Launcher for optimization test. Choose which to run via command line args:',
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

    ########################################################################
    # Run the optimization tests called for by the user ####################
    ########################################################################

    # Create multiprocessing queue so we can run each optimization test
    # in it's own subprocess. This makes sure that any artifacts, e.g. CUDA 
    # contexts, die with the test that created them and don't interfere 
    # with subsequent tests etc.

    queue = Queue()

    #### Run device map strategy optimization ##############################

    if args.device_map == 'True' or args.run_all == 'True':

        p = Process(target=device_map.optimize,
            kwargs=dict(
                resume=args.resume,
                output_dir=f'{conf.OPTIMIZATION_DIR}/device_map',
                output_filename='results.csv',
                replicates=10,
                device_map_strategies=[
                    'CPU only',
                    'multi-GPU',
                    'single GPU',
                    'balanced',
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

    #### Run batch size optimization #######################################
        
    if args.batch_size == 'True' or args.run_all == 'True':

        p = Process(target=batch_size.optimize,
            kwargs=dict(
                resume=args.resume,
                output_dir=f'{conf.OPTIMIZATION_DIR}/batch_size',
                output_filename='results.csv',
                replicates=10,
                batches=3,
                db_name=conf.DB_NAME,
                user=conf.USER,
                passwd=conf.PASSWD,
                host=conf.HOST
            )
        )

        p.start()
        p.join()