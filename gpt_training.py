# Step 1: Install required libraries (only needed once)
#pip install transformers datasets

# Step 2: Import libraries
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

# Step 3: Load dataset (you can replace this with your own text)
dataset = load_dataset("https://indianexpress.com/section/technology/artificial-intelligence/")
print(dataset)

# Step 4: Load GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # GPT2 doesn't have a pad token
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Step 5: Tokenize the text
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Step 6: Create data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Step 7: Set training arguments
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",
    overwrite_output_dir=True,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    fp16=True,  # Enable if on supported GPU
)

# Step 8: Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Step 9: Train the model
trainer.train()

# Step 10: Generate text
from transformers import pipeline

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
print(generator("Once upon a time", max_length=50, num_return_sequences=1))
