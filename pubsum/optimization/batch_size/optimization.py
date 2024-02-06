import time
import torch
import optimization.classes as opt_classes
import optimization.helper_functions as helper_funcs

########################################################################
def optimize(
    resume: str,
    output_dir: str,
    output_filename: str,
    replicates: int,
    batches: int,
    db_name: str,
    user: str,
    passwd: str,
    host: str
) -> bool:
    
    '''Uses loop to find largest batch size that will run.'''
    print(f'\nRunning batch size optimization with resume={resume}')

    # Start the LLM and tokenizer
    summarizer=opt_classes.Summarizer()

    # Instantiate results object and resume data collection if asked
    results=opt_classes.Results(
        independent_vars=['replicate', 'batch size'],
        dependent_vars=[
            'summarization rate (abstracts/sec.)',
            'max memory allocated (bytes)'
        ],
        output_dir=output_dir, 
        output_filename=output_filename,
        resume=resume
    )

    # If we are resuming, and have old data, get the largest 
    # completed batch size and start there
    if resume == 'True' and len(results.completed_runs) > 0:
        completed_batch_sizes=[completed_run[1] for completed_run in results.completed_runs]
        batch_size=max(completed_batch_sizes)

    # If we are not resuming, start batch size at zero
    else:
        batch_size=0

    # Loop with increasing batch size until we get an OOM
    while True:
        batch_size+=1
        
        # Loop on replicates for this batch size
        for replicate in range(1, replicates+1):

            print(f'\nBatch size optimization')
            print(f' Batch size: {batch_size}')
            print(f' Replicate: {replicate} of {replicates}')

            # Fence to catch out of memory errors and return
            try:
                # Get enough rows to make batches from abstracts table
                rows=helper_funcs.get_rows(
                    db_name=db_name, 
                    user=user, 
                    passwd=passwd, 
                    host=host, 
                    num_abstracts=batches*batch_size
                )

                # Pre-batch abstracts for this replicate
                batched_abstracts=[]

                for i in range(batches):

                    # Get the batch and then the abstracts from that batch
                    batch=rows[i*batch_size:(i+1)*batch_size]
                    abstracts=[row[1] for row in batch]
                    batched_abstracts.append(abstracts)

                # Start and time the batch loop
                summarization_start=time.time()

                for i in range(batches):

                    # Do the summary
                    _=summarizer.summarize(batched_abstracts[i])

                # Stop the clock
                dT=time.time() - summarization_start

                # Get max memory
                max_memory=torch.cuda.max_memory_allocated()

                # Collect run results
                results.data['replicate'].append(
                    int(replicate)
                )
                results.data['batch size'].append(
                    int(batch_size)
                )
                results.data['max memory allocated (bytes)'].append(
                    int(max_memory)
                )
                results.data['summarization rate (abstracts/sec.)'].append(
                    (batches*batch_size)/dT
                )

            # Catch out of memory errors
            except torch.cuda.OutOfMemoryError:
                print('\nDied on OOM - done\n')
                
                # Everyone can go home
                return True

            # Reset memory tracking
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # After all replicates of this batch size are complete, save the results
        # and clear the results buffer for the next run
        results.save_results()
        results.clear_results()

    return True