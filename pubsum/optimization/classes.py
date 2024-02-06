import os
import gc
import torch
import transformers
import pandas as pd
from typing import List
import optimization.helper_functions as helper_funcs

########################################################################
class Results:

    '''Class to hold objects and methods for collection of results.'''

    def __init__(
        self,
        independent_vars: List[str],
        dependent_vars: List[str],
        output_dir: str,  
        output_filename: str,
        resume: bool,
    ):

        # Add variable keys
        self.independent_vars=independent_vars
        self.dependent_vars=dependent_vars

        # Output file for results
        self.output_file=f'{output_dir}/{output_filename}'

        # Create dict of empty lists for data collection
        self.data={}
        for var in self.independent_vars+self.dependent_vars:
            self.data[var]=[]

        # Empty list for tuples of values of independent variables for
        # completed runs, used to make sure we are not repeating runs
        # when/if resuming
        self.completed_runs=[]

        # Read completed runs from old data if asked
        if resume == 'True':

            # Read existing results if any
            if os.path.exists(self.output_file):
                old_results_df=pd.read_csv(self.output_file)

                # Loop on old results and collect values 
                # of independent vars
                old_results=[]
                for var in self.independent_vars:
                    old_results.append(old_results_df[var].to_list())

                # Zip the lists of old data into tuples
                self.completed_runs=list(zip(*old_results))
                print(f'Resuming data collection with {len(self.completed_runs)} runs complete')

        # If we are not resuming, write empty file
        elif resume == 'False':
            self.save_results(overwrite=True)

    def save_results(self, overwrite: bool=False) -> None:

        '''Persist current contents of self.data to disk. Overwrite any 
        data in self.output_file by default or append if requested.'''

        # Make dataframe of new results
        results_df = pd.DataFrame(self.data)

        if overwrite == False:

            # Read existing results, if any and concatenate new results
            if os.path.exists(self.output_file):
                old_results_df = pd.read_csv(self.output_file)
                results_df = pd.concat([old_results_df, results_df])

        else:
            print('Clearing any old results')

        # Save results for run to csv
        results_df.to_csv(self.output_file, index=False)

    def clear_results(self) -> None:

        '''Empties contents of self.data.'''

        # Assign empty list for all collection vars
        for var in self.independent_vars+self.dependent_vars:
            self.data[var]=[]

########################################################################
class Summarizer:

    '''Objects and methods related to LLM for summarization.'''

    def __init__(
        self,
        device_map_strategy: str='single GPU',
        model_name: str='haining/scientific_abstract_simplification',
        generation_max_length: int=256,
        generation_top_p: float=0.9,
        generation_do_sample: bool=True,
        tokenizer_max_length: int=672, 
        tokenizer_padding: str='max_length', 
        tokenizer_truncation: bool=True, 
        tokenizer_return_tensors: str='pt'
    ):

        # Parse device map strategy into huggingface device map string
        self.device_map=helper_funcs.translate_device_map_strategy(
            device_map_strategy
        )

        # Determine if we need to use GPU based on device map strategy
        self.gpu_run=helper_funcs.use_gpu(device_map_strategy)

        # Fire up model
        self.model=transformers.AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            device_map=self.device_map
        )

        # Load generation config from model 
        self.gen_cfg=transformers.GenerationConfig.from_model_config(
            self.model.config
        )

        # Set some generation hyperparameters
        self.gen_cfg.max_length=generation_max_length
        self.gen_cfg.top_p=generation_top_p
        self.gen_cfg.do_sample=generation_do_sample

        # Initialize the tokenizer
        self.tokenizer=transformers.AutoTokenizer.from_pretrained(
            model_name
        )

        # Set some tokenizer parameters 
        self.tokenizer_max_length=tokenizer_max_length 
        self.tokenizer_padding=tokenizer_padding
        self.tokenizer_truncation=tokenizer_truncation
        self.tokenizer_return_tensors=tokenizer_return_tensors

    def summarize(self, inputs: List[str]) -> List[str]:

        '''Takes a list of texts and summarizes them.'''

        # Prepend the standard prompt to each input
        inputs=['summarize, simplify, and contextualize: '+input for input in inputs]

        # Encode the inputs
        encoding = self.tokenizer(
            inputs, 
            max_length=self.tokenizer_max_length,
            padding=self.tokenizer_padding,
            truncation=self.tokenizer_truncation,
            return_tensors=self.tokenizer_return_tensors
        )

        # Move to GPU, if needed
        if self.gpu_run == True:
            encoding=encoding.to('cuda')
        
        # Generate summary
        decoded_ids=self.model.generate(
            input_ids=encoding['input_ids'],
            attention_mask=encoding['attention_mask'], 
            generation_config=self.gen_cfg
        )
        
        # Decode summaries
        summaries=self.tokenizer.decode(decoded_ids[0], skip_special_tokens=True)

        return summaries


