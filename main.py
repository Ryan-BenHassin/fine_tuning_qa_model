from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer, AutoConfig
from datasets import load_dataset
from myFunctions import preprocess_function
from transformers import DefaultDataCollator
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
np.object = object; np.int = int; np.bool = bool; np.float = float

# ___________ Data preprocessing ____________________________________________

squad = load_dataset("Ryan20/qa_hotel_dataset_2", split="train")

print(squad)

model_name= "bert-base-cased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

# tokenized_squad = squad.map(
#     preprocess_function, 
#     batched=True, 
#     remove_columns=squad.data.column_names
# )

data_collator = DefaultDataCollator()

config = AutoConfig.from_pretrained("bert-base-cased")
config.max_position_embeddings = 1000 # match tokenized input size

# # ___________ Model training ____________________________________________
# model = AutoModelForQuestionAnswering.from_pretrained(model_name, config=config, ignore_mismatched_sizes=True)

# training_args = TrainingArguments(
#     output_dir="sqoin_qa_model_first",
#     evaluation_strategy="epoch",
#     learning_rate=2e-5,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     num_train_epochs=3,
#     weight_decay=0.01,
#     push_to_hub=True,
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_squad,
#     eval_dataset=tokenized_squad,
#     tokenizer=tokenizer,
#     data_collator=data_collator,
# )

# trainer.train()

# trainer.push_to_hub()