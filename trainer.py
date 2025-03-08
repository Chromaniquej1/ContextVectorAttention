from transformers import TrainingArguments, Trainer
import torch
import time
from thop import profile

def calculate_latency_per_image(model, data_loader, device):
    """Calculate latency per image."""
    model.eval()
    total_time = 0
    total_images = 0

    with torch.no_grad():
        for batch in data_loader:
            inputs = batch['x'].to(device)
            batch_size = inputs.shape[0]
            total_images += batch_size

            start_time = time.time()
            _ = model(inputs)
            end_time = time.time()

            total_time += (end_time - start_time)

    avg_latency_per_image = (total_time / total_images) * 1000  # Convert to milliseconds
    return avg_latency_per_image

def calculate_latency_per_batch(model, data_loader, device):
    """Calculate latency per batch."""
    model.eval()
    total_time = 0
    num_batches = 0

    with torch.no_grad():
        for batch in data_loader:
            inputs = batch['x'].to(device)
            start_time = time.time()
            _ = model(inputs)
            total_time += time.time() - start_time
            num_batches += 1

    avg_latency = total_time / num_batches
    return avg_latency

def calculate_throughput(model, data_loader, device):
    """Calculate throughput in samples per second."""
    model.eval()
    total_samples = 0
    total_time = 0

    with torch.no_grad():
        for batch in data_loader:
            inputs = batch['x'].to(device)
            batch_size = inputs.size(0)
            start_time = time.time()
            _ = model(inputs)
            total_time += time.time() - start_time
            total_samples += batch_size

    throughput = total_samples / total_time
    return throughput

def count_model_parameters(model):
    """Count the number of trainable parameters."""
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params / 1_000_000