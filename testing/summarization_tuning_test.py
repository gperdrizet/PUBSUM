import datetime
import evaluate
import pandas as pd
import numpy as np
from copy import deepcopy
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, TrainerCallback

'''Fine tuning of summarization on California subset of the BillSum dataset.
from: https://huggingface.co/docs/transformers/tasks/summarization

GOTCHAs

1. You're using a T5TokenizerFast tokenizer. Please note that with a fast 
tokenizer, using the `__call__` method is faster than using a method to 
encode the text followed by a call to the `pad` method to get a padded encoding.

See here for explanation: https://github.com/huggingface/transformers/issues/22638

Solution is to add:

tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

After the tokenizer to turn the warning off.

2. torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 512.00 MiB 
(GPU 0; 11.17 GiB total capacity; 10.55 GiB already allocated; 142.25 MiB free; 
10.58 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory 
try setting max_split_size_mb to avoid fragmentation.  See documentation for 
Memory Management and PYTORCH_CUDA_ALLOC_CONF.

Solution 1: change batch size from 16 to 8 - max memory use ~ 9 GB, runs.

3. huggingface_transformers/.env/lib/python3.8/site-packages/torch/nn/parallel/_functions.py:68: 
UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; 
will instead unsqueeze and return a vector.

See here: https://github.com/huggingface/transformers/issues/14128

Setting os.environ["CUDA_VISIBLE_DEVICES"] = '0' causes a bunch of even worse errors and 
doesn't run. So no insight on wether this is a multi-GPU issue or not.

The other solution about DDP seems like overkill for this test example.
Setting fp16 to false also does not solve it.

Since it's running and rouge is improving, I think I will leave it alone for now.

'''

def preprocess_function(examples):
    '''Takes a prefix and training examples, prepends prefix to text and tokenizes,
    text and summary. Returns dict containing result.'''

    prefix = "summarize: "

    # Prepend prefix to each text
    inputs = [prefix + doc for doc in examples["text"]]

    # Tokenize text with truncation
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    # Tokenize summary
    labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)

    # Add labels to result
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

def compute_metrics(eval_pred):
    '''Passes predictions and true labels to compute, returns rouge metric'''

    # Expand predictions and labels
    predictions, labels = eval_pred

    # Un-tokenize predictions
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Un-tokenize labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute rouge score
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    # Get mean prediction length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    # Return rounded result
    return {k: round(v, 4) for k, v in result.items()}

class CustomCallback(TrainerCallback):
    
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy

# Load and train test split dataset
billsum = load_dataset("billsum", split = "ca_test")
billsum = billsum.train_test_split(test_size = 0.2)

print(f'\nbillsum type: {type(billsum)}')
print(f'billsum keys: {billsum.keys()}\n')
print(f'billsum key train type: {type(billsum["train"])}')
print(f'billsum key train len: {len(billsum["train"])}\n')
print(f'Train element type: {type(billsum["train"][0])}')
print(f'Train element keys: {billsum["train"][0].keys()}\n')

for key, val in billsum["train"][0].items():
    print(f' Train {key} is: {type(val)}')

print()

# Load tokenizer
checkpoint = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
print(f'Tokenizer is: {type(tokenizer)}\n')

# Add prefix prompt and tokenize training data
tokenized_billsum = billsum.map(preprocess_function, batched = True)
print(f'\nTokenized billsum is: {type(tokenized_billsum)}')
print(f'Tokenized billsum keys: {tokenized_billsum.keys()}')
print(f'Tokenized billsum train is {type(tokenized_billsum["train"])}\n')

# Create padded batch of examples
data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer, model = checkpoint)
print(f'Data collator is: {type(data_collator)}')

# Load up evaluation function
rouge = evaluate.load("rouge")
print(f'Rouge evaluation function is: {type(rouge)}')

# Load up the model from the checkpoint
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
print(f'Model is: {type(model)}\n')

training_args = Seq2SeqTrainingArguments(
    output_dir = "./.cache/models--t5-billsum",
    evaluation_strategy = "epoch",
    #eval_steps = 1,
    learning_rate = 2e-5,
    per_device_train_batch_size = 8,
    per_device_eval_batch_size = 8,
    weight_decay = 0.005,
    save_total_limit = 1,
    num_train_epochs = 50,
    predict_with_generate = True,
    fp16 = True,
    push_to_hub = False,
    load_best_model_at_end=True,
    save_strategy = "epoch"
)

trainer = Seq2SeqTrainer(
    model = model,
    args = training_args,
    train_dataset = tokenized_billsum["train"],
    eval_dataset = tokenized_billsum["test"],
    tokenizer = tokenizer,
    data_collator = data_collator,
    compute_metrics = compute_metrics,
)

trainer.add_callback(CustomCallback(trainer))
result_date = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M')
results = trainer.train()
history = trainer.state.log_history

print(f'\nTrainer:\n\n{dir(trainer)}\n')
print(f'Results:\n\n{dir(results)}\n')
print(f'History:\n\n{dir(history)}\n')
print(f'History is: {type(history)}')

training_results = []
evaluation_results = []

for epoch in history[:-1]:
    print(f'\n{epoch}')

    if 'eval_loss' in epoch.keys():
        evaluation_results.append(epoch)
    
    elif 'train_loss' in epoch.keys():
        training_results.append(epoch)
    
print()

training_results_df = pd.DataFrame(training_results)
evaluation_results_df = pd.DataFrame(evaluation_results)

results_df = pd.merge(
    training_results_df,
    evaluation_results_df,
    how = "inner",
    on = ['epoch', 'step'],
    left_index = False,
    right_index = False,
)

print(results_df)
print()
print(results_df.info())

result_file = f'./testing/training_run_results/{result_date}.csv'
results_df.to_csv(result_file, index=False)