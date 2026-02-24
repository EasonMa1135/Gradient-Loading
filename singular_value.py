import os
import torch
import csv

# ================= Configuration =================
FILE_PATH = "./gradient_checkpoints/grads_step_100.pt"  # Replace with the path to your .pt file
OUTPUT_CSV = "all_layers_singular_values.csv"

if not os.path.exists(FILE_PATH):
    print(f"Error: File not found at {FILE_PATH}")
    exit()

print(f"Loading gradient file: {FILE_PATH}")
# map_location='cpu' ensures it runs without a GPU, weights_only=True is a safety best practice
grads = torch.load(FILE_PATH, map_location='cpu', weights_only=True)

# Lists to store the results
results = []
skipped_layers = []

print("\nStarting Singular Value Decomposition (SVD) for each layer...")

for layer_name, grad_tensor in grads.items():
    # 1. Filter: SVD can only be applied to 2D tensors (matrices)
    # Biases and LayerNorm weights/biases in GPT-2 are 1D and do not have singular values
    if grad_tensor.dim() != 2:
        skipped_layers.append((layer_name, list(grad_tensor.shape)))
        continue
        
    # 2. Convert data type to float32 to ensure numerical stability during SVD
    grad_matrix = grad_tensor.float()
    
    # 3. Calculate singular values
    # svdvals returns a 1D tensor, sorted in descending order automatically
    singular_values = torch.linalg.svdvals(grad_matrix)
    
    # 4. Extract key metrics
    # We extract: Matrix Shape, Max SV (Top 1), 2nd SV, 3rd SV, and Min SV
    sv_list = singular_values.tolist()
    max_sv = sv_list[0]
    sv_2 = sv_list[1] if len(sv_list) > 1 else None
    sv_3 = sv_list[2] if len(sv_list) > 2 else None
    min_sv = sv_list[-1]
    
    # Calculate Condition Number = Max SV / Min SV
    # A larger condition number indicates an ill-conditioned matrix, making optimization harder
    condition_number = max_sv / (min_sv + 1e-8)  # Add a tiny epsilon to prevent division by zero
    
    # Save data for this layer
    results.append([
        layer_name, 
        f"{grad_matrix.shape[0]}x{grad_matrix.shape[1]}",
        max_sv, 
        sv_2, 
        sv_3, 
        min_sv, 
        condition_number
    ])
    
    print(f"Processed: {layer_name} | Max SV: {max_sv:.4f}")

# ================= Export Data =================
print(f"\nExporting results to {OUTPUT_CSV}...")
with open(OUTPUT_CSV, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write header
    writer.writerow(["Layer Name", "Shape", "Max_SV (Spectral Norm)", "SV_2", "SV_3", "Min_SV", "Condition Number"])
    # Write data rows
    writer.writerows(results)

print("\n--- Summary Statistics ---")
print(f"Successfully calculated singular values for {len(results)} matrices.")
print(f"Skipped {len(skipped_layers)} non-matrix tensors (e.g., biases or LayerNorm weights).")
print(f"Detailed report saved to: {OUTPUT_CSV}")