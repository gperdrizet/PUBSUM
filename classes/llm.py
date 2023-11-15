import config as conf
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig

class Llm:
    '''Class to hold object related to the llm'''

    def __init__(self):

        self.use_gpu = conf.use_gpu
        self.num_jobs = conf.num_jobs

        # Initialize LLM for summarization
        self.model = AutoModelForSeq2SeqLM.from_pretrained("haining/scientific_abstract_simplification")
        
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
        
        # Generate summary
        decoded_ids = self.model.generate(
            input_ids = encoding['input_ids'],
            attention_mask = encoding['attention_mask'], 
            generation_config = self.gen_cfg
        )
        
        # Decode summary
        summary = self.tokenizer.decode(decoded_ids[0], skip_special_tokens = True)

        return summary