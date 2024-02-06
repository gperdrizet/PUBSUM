import time
import itertools
from typing import List
import optimization.classes as opt_classes
import optimization.helper_functions as helper_funcs

########################################################################
def optimize(
    resume: bool, 
    output_dir: str,
    output_filename: str,
    replicates: int,
    device_map_strategies: List[str],
    db_name: str,
    user: str,
    passwd: str,
    host: str
):
    
    '''Try huggingface device maps to find the fastest one.'''
    print(f'\nRunning device map optimization with resume={resume}')

    # Start the LLM and tokenizer
    summarizer = opt_classes.Summarizer()

    # Instantiate results object and resume data collection if asked
    results=opt_classes.Results(
        independent_vars=['device map strategy', 'replicate'],
        dependent_vars=['summarization rate (abstracts/sec.)'],
        output_dir=output_dir, 
        output_filename=output_filename,
        resume=resume
    )

    # If the resumed results already contain all of the runs we are
    # going to do, we are done and everyone can go home
    replicate_numbers = list(range(1, replicates + 1))
    total_runs = len(device_map_strategies) * len(replicate_numbers)

    if total_runs == len(results.completed_runs):
        print('Optimization is complete')
        return True

    # Otherwise, construct parameter sets for run
    all_parameter_sets = itertools.product(
        device_map_strategies, 
        replicate_numbers
    )

    # Keep only those which have not already been completed
    parameter_sets=[]
    for parameter_set in all_parameter_sets:
        if parameter_set not in results.completed_runs:
            parameter_sets.append(parameter_set)
        
    # Start the run - main loop is on sets of independent variables
    for parameter_set in parameter_sets:

        # Unpack parameters from set
        replicate=parameter_set[1]
        device_map_strategy=parameter_set[0]
        device_map=helper_funcs.translate_device_map_strategy(device_map_strategy)

        print(f'\nDevice map strategy optimization')
        print(f' Device map strategy: {device_map_strategy}')
        print(f' Replicate: {replicate} of {replicates}')

        # Check if we need to restart the model with
        # a new device map for this run
        if summarizer.device_map != device_map:
            summarizer.clear()
            summarizer=opt_classes.Summarizer(device_map_strategy=device_map_strategy)

        # Get abstract from row in abstracts table
        row=helper_funcs.get_rows(
            db_name=db_name, 
            user=user, 
            passwd=passwd, 
            host=host, 
            num_abstracts=1
        )

        abstract=[row[0][1]]

        # Do and time the summary
        summarization_start = time.time()
        _=summarizer.summarize(abstract)
        dT=time.time() - summarization_start

        # Collect results
        results.data['replicate'].append(replicate)
        results.data['device map strategy'].append(device_map_strategy)
        results.data['summarization rate (abstracts/sec.)'].append(1/dT)

        # Save the result and reinitialize collector after each set
        # of replicates is complete
        if replicate % replicates == 0:
            results.save_results()
            results.clear_results()

    print('\nDone\n')
    return True