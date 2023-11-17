import config as conf
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig

class Llm:
    '''Class to hold object related to the llm'''

    def __init__(self, device_map_strategy):

        self.device_map_strategy = device_map_strategy

        # Translate device map strategy for huggingface. Start with
        # device map set to CPU only by default
        device_map = 'cpu'

        if device_map_strategy == 'multi-GPU':
            device_map = 'auto'

        elif device_map_strategy == 'single GPU':
            device_map = 'cuda:0'

        elif device_map_strategy == 'balanced':
            device_map = 'balanced'

        elif device_map_strategy == 'balanced_low_0':
            device_map = 'balanced_low_0'

        elif device_map_strategy == 'sequential':
            device_map = 'sequential'

        self.model = AutoModelForSeq2SeqLM.from_pretrained("haining/scientific_abstract_simplification", device_map = device_map)
        
        # Load generation config from model and set some parameters as desired
        self.gen_cfg = GenerationConfig.from_model_config(self.model.config)
        self.gen_cfg.max_length = 256
        self.gen_cfg.top_p = 0.9
        self.gen_cfg.do_sample = True

        # Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("haining/scientific_abstract_simplification")

        # Set prompt to prepend to abstracts
        self.instruction = "summarize, simplify, and contextualize: "

    def summarize(self, abstract):

        # Prepend the prompt to this abstract and encode
        encoding = self.tokenizer(
            self.instruction + abstract, 
            max_length = 672, 
            padding = 'max_length', 
            truncation = True, 
            return_tensors = 'pt'
        )

        # Move to GPU if appropriate
        if self.device_map_strategy != 'CPU only':
            encoding = encoding.to('cuda')
        
        # Generate summary
        decoded_ids = self.model.generate(
            input_ids = encoding['input_ids'],
            attention_mask = encoding['attention_mask'], 
            generation_config = self.gen_cfg
        )
        
        # Decode summary
        summary = self.tokenizer.decode(decoded_ids[0], skip_special_tokens = True)

        return summary