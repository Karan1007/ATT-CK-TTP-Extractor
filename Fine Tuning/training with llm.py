


# -------------------- IMPORTS --------------------
import os
import logging
import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments,
    Trainer, DataCollatorForSeq2Seq
)

# -------------------- SETUP --------------------
# ✅ Disable WandB (Weights & Biases) to prevent logging issues
os.environ["WANDB_DISABLED"] = "true"

# ✅ Force CPU usage (if no GPU available)
device = torch.device("cpu")

# ✅ Set logging level
logging.basicConfig(level=logging.INFO)

# -------------------- LOAD DATASET --------------------
logging.info("🔹 Loading dataset...")
dataset = load_dataset("json", data_files="C:/Users/karki/OneDrive/Desktop/generative AI/fine_tune_data_fixed.jsonl")

# ✅ Split dataset (90% train, 10% validation)
train_test_split = dataset["train"].train_test_split(test_size=0.1)
train_data = train_test_split["train"]
valid_data = train_test_split["test"]

logging.info(f"✅ Dataset Loaded! Train Size: {len(train_data)}, Validation Size: {len(valid_data)}")

# -------------------- LOAD MODEL & TOKENIZER --------------------
logging.info("🔹 Loading model & tokenizer...")

# ✅ Use a lightweight model (best for CPU)
model_name = "EleutherAI/gpt-neo-125M"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ✅ Fix: Assign a padding token (GPT models often don't have one)
tokenizer.pad_token = tokenizer.eos_token  # Set EOS token as padding

model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

logging.info("✅ Model & Tokenizer Loaded Successfully!")


# -------------------- TOKENIZATION FUNCTION --------------------

from datasets import DatasetDict

# Define tokenization function
def tokenize_function(examples):
    user_messages = [msg["content"] for entry in examples["messages"] for msg in entry if msg["role"] == "user"]
    assistant_messages = [msg["content"] for entry in examples["messages"] for msg in entry if msg["role"] == "assistant"]

    inputs = tokenizer(user_messages, padding="max_length", truncation=True, max_length=512)
    labels = tokenizer(assistant_messages, padding="max_length", truncation=True, max_length=512)

    inputs["labels"] = labels["input_ids"]
    return inputs

# Apply tokenization to training and validation sets
tokenized_train = train_data.map(tokenize_function, batched=True)
tokenized_valid = valid_data.map(tokenize_function, batched=True)


logging.info("✅ Tokenization complete!")

# -------------------- TRAINING ARGUMENTS --------------------
training_args = TrainingArguments(
    output_dir="./tinyllama-finetuned",  # Save model locally
    per_device_train_batch_size=1,  # ✅ Lower batch size for CPU
    per_device_eval_batch_size=1,
    num_train_epochs=1,  # ✅ Reduce epochs for faster training
    save_total_limit=1,  # ✅ Keep only the latest checkpoint
    save_strategy="epoch",
    evaluation_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    report_to="none",  # ✅ No WandB logging
    push_to_hub=False,  # ✅ Avoid Hugging Face Hub uploads
    load_best_model_at_end=True  # ✅ Auto-load best model
)

logging.info("✅ Training Arguments Set!")

# -------------------- FINE-TUNING --------------------
# ✅ Data collator for better padding
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# ✅ Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# ✅ Start training
logging.info("🔹 Starting fine-tuning...")
try:
    trainer.train()
    logging.info("✅ Fine-tuning Complete! Model is now trained on MITRE ATT&CK reports.")
except Exception as e:
    logging.error(f"🚨 Training crashed! Error: {str(e)}")

# -------------------- SAVE MODEL --------------------
# ✅ Save model & tokenizer locally
model.save_pretrained("tinyllama-mitre")
tokenizer.save_pretrained("tinyllama-mitre")

logging.info("✅ Model Saved Successfully!")
