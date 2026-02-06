import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW

MODEL_NAME = "gpt2-medium"
TRAIN_FROM_SCRATCH = False 

BATCH_SIZE = 4      
SEQ_LENGTH = 512 
LEARNING_RATE = 5e-5
SAVE_INTERVAL = 100
MAX_STEPS = 1000
OUTPUT_DIR = "./gradient_checkpoints"

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {device}")

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

if TRAIN_FROM_SCRATCH:
    print("Initializing random weights (Training from scratch)...")
    config = GPT2Config.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel(config).to(device)
else:
    print(f"Loading pre-trained {MODEL_NAME} weights...")
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(device)

print("Loading FineWeb dataset stream...")
dataset = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)

def collate_fn(batch_items):
    texts = [item['text'] for item in batch_items]
    inputs = tokenizer(
        texts, 
        return_tensors="pt", 
        max_length=SEQ_LENGTH, 
        truncation=True, 
        padding="max_length"
    )
    return inputs.input_ids.to(device), inputs.attention_mask.to(device)

data_iter = iter(dataset)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

print("Starting training loop...")
model.train()

for step in range(MAX_STEPS):
    batch_texts = []
    try:
        for _ in range(BATCH_SIZE):
            batch_texts.append(next(data_iter))
    except StopIteration:
        print("Dataset exhausted.")
        break
        
    input_ids, attention_mask = collate_fn(batch_texts)
    
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    loss = outputs.loss
    
    optimizer.zero_grad()
    loss.backward()
    
    if step % SAVE_INTERVAL == 0:
        print(f"Step {step}: Loss = {loss.item():.4f}. Saving gradients...")
        
        grads_to_save = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grads_to_save[name] = param.grad.clone().half().cpu()
        
        save_path = os.path.join(OUTPUT_DIR, f"grads_step_{step}.pt")
        torch.save(grads_to_save, save_path)
        print(f"Saved: {save_path}")

    optimizer.step()

print("Done.")