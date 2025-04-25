"""
Stateful training script for QwenWithIntegratedMemory.

This script demonstrates training on continuous sequences broken into fixed-size segments,
with neural memory state persisting across segments within each sequence.
"""
import gzip
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from adam_atan2_pytorch import AdoptAtan2
from qwen_with_memory import create_qwen_with_memory
import wandb

# Configuration
SEGMENT_LEN = 512               # Length of each segment
NUM_SEGMENTS = 4                # Number of segments per sequence
MAX_CONTEXT_MULTIPLIER = 3      # Maximum context window as a multiple of segment length
TOTAL_SEQ_LEN = SEGMENT_LEN * NUM_SEGMENTS  # Total length of sequence processed statefully
BATCH_SIZE = 4
NUM_BATCHES = 10000
LEARNING_RATE = 2e-4
GRAD_CLIP = 0.5
QWEN_BASE_MODEL = "Qwen/Qwen2.5-0.5B"

# Initialize Weights & Biases
wandb.init(project="titans-qwen-memory", mode="disabled")
wandb.run.name = f"stateful - {NUM_SEGMENTS} segments - context {MAX_CONTEXT_MULTIPLIER}x"

# Load enwik8 data
with gzip.open('data/enwik8.gz', 'rb') as f:
    data = np.frombuffer(f.read(int(95e6)), dtype=np.uint8).copy()
    data_train, data_val = np.split(data, [int(90e6)])
    data_train = torch.from_numpy(data_train).long()

class StatefulTextDataset(Dataset):
    """
    Dataset of contiguous sequences for stateful training.
    Each example is a slice of the raw data of length TOTAL_SEQ_LEN + 1,
    so we can form inputs and targets across segment boundaries.
    """
    def __init__(self, data, total_len):
        super().__init__()
        self.data = data
        self.total_len = total_len

    def __len__(self):
        return (self.data.size(0) - (self.total_len + 1)) // self.total_len

    def __getitem__(self, idx):
        start = idx * self.total_len
        seq = self.data[start:start + self.total_len + 1]
        return seq

# Prepare dataset and loader
train_dataset = StatefulTextDataset(data_train, TOTAL_SEQ_LEN)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
)

# Instantiate model and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define memory configuration for Qwen model
memory_config = {
    "hidden_size": 896,  # Must match the hidden size of QWEN_BASE_MODEL
    "segment_len": SEGMENT_LEN,
    "num_longterm_mem_tokens": 4,
    "num_persist_mem_tokens": 0,
    "neural_memory_layers": (2, 4, 6),
    "neural_memory_segment_len": SEGMENT_LEN + 4,
    "neural_mem_gate_attn_output": False,
    "neural_memory_qkv_receives_diff_views": True,
    "neural_mem_weight_residual": False,
    "dim_head": 64,
    "heads": 8,
    "memory_depth": 2,
    "max_context_multiplier": MAX_CONTEXT_MULTIPLIER,  # Limit context to N * segment_len
    "use_limited_context": True  # Enable context limitation
}

# Create the Qwen model with memory and context limitation
model = create_qwen_with_memory(
    model_name_or_path=QWEN_BASE_MODEL,
    memory_config=memory_config
).to(device)

optim = AdoptAtan2(model.parameters(), lr=LEARNING_RATE)

# Training Loop: stateful over NUM_SEGMENTS segments per sequence
for batch_idx, batch_seq in enumerate(train_loader):
    if batch_idx >= NUM_BATCHES:
        break

    batch_seq = batch_seq.to(device)
    inputs = batch_seq[:, :-1]    # shape: [BATCH_SIZE, TOTAL_SEQ_LEN]
    targets = batch_seq[:, 1:]

    # Reshape into segments: [BATCH_SIZE, NUM_SEGMENTS, SEGMENT_LEN]
    inputs = inputs.view(BATCH_SIZE, NUM_SEGMENTS, SEGMENT_LEN)
    targets = targets.view(BATCH_SIZE, NUM_SEGMENTS, SEGMENT_LEN)

    # No need to explicitly reset memory state - will be handled via reset_memory param
    optim.zero_grad()

    total_loss = 0.0
    # Process each segment sequentially, keeping state
    for seg_idx in range(NUM_SEGMENTS):
        x = inputs[:, seg_idx, :]
        y = targets[:, seg_idx, :]
        
        # First segment resets memory (reset_memory=True), subsequent ones don't (reset_memory=False)
        reset_mem = (seg_idx == 0)
        
        # Pass input_ids=x and labels=y
        out = model(input_ids=x, labels=y, reset_memory=reset_mem, return_dict=True)
        loss = out.loss
        total_loss += loss.item()
        
        # Backprop scaled by number of segments
        (loss / NUM_SEGMENTS).backward()

    # Gradient clipping and optimization step
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optim.step()
    optim.zero_grad()

    # Logging
    avg_loss = total_loss / NUM_SEGMENTS
    wandb.log({ 'loss': avg_loss, 'batch': batch_idx })
    if batch_idx % 100 == 0:
        print(f"Batch {batch_idx}, Avg Loss: {avg_loss:.4f}")
