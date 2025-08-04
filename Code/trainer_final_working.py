import torch
import networkx as nx
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from datetime import datetime
from tqdm import tqdm
from joblib import Parallel, delayed
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.nn import GRU

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the main weather dataset
weather_csv_path = r"C:\Users\LENOVO\Desktop\semproject\merged_weather_locations_cleaned_2.csv"  # Update with your weather CSV path
print("Loading weather data...")
weather_df = pd.read_csv(weather_csv_path)

# Convert date to datetime and ensure correct format
weather_df['date'] = pd.to_datetime(weather_df['date'], format='%Y-%m-%d')  # Ensure consistent datetime format
weather_df = weather_df.sort_values(by='date')  # Sort by date to ensure chronological order

# Add temporal features (day of the year, month)
weather_df['day_of_year'] = weather_df['date'].dt.dayofyear
weather_df['month'] = weather_df['date'].dt.month

# Step 1: Preprocess the Graph
file_path = r"C:\Users\LENOVO\Desktop\semproject\weather_graph_with_deltas.gexf"
print("Loading graph...")
G = nx.read_gexf(file_path)

# Debug: Check if the graph has edges
print(f"Number of edges in the graph: {G.number_of_edges()}")
if G.number_of_edges() == 0:
    raise ValueError("The graph has no edges. Please check the graph file.")

# Normalize location names in the graph
print("Normalizing graph locations...")
for node, attrs in tqdm(G.nodes(data=True), desc="Normalizing nodes"):
    attrs['location'] = attrs['location'].strip().lower()

# Normalize location names in the weather data
weather_df['Location_x'] = weather_df['Location_x'].str.strip().str.lower()

# Create a mapping from location to node ID
location_to_node = {attrs['location']: node for node, attrs in G.nodes(data=True)}

# Debug: Check if location_to_node is populated
print(f"Number of locations mapped to nodes: {len(location_to_node)}")
if len(location_to_node) == 0:
    raise ValueError("No locations were mapped to nodes. Check the graph and weather data.")

# Create a mapping from node IDs to their attributes
node_to_attrs = {node: attrs for node, attrs in G.nodes(data=True)}

# Step 2: Prepare Node Features and Targets
node_indices = {}  # Map node IDs to indices in the feature/target tensors

# Create a set of unique locations from the weather data
unique_locations = weather_df['Location_x'].unique()

# Populate node_indices with unique locations
print("Mapping locations to nodes...")
for location in tqdm(unique_locations, desc="Mapping locations"):
    if location in location_to_node:
        node_id = location_to_node[location]
        if node_id not in node_indices:
            node_indices[node_id] = len(node_indices)
    else:
        print(f"Warning: Location '{location}' in weather data is not mapped to a node in the graph.")

# Debug: Check if node_indices is populated correctly
print(f"Number of unique locations mapped to node indices: {len(node_indices)}")
if len(node_indices) == 0:
    raise ValueError("No locations were mapped to nodes. Check the graph and weather data.")

# Define num_nodes globally
num_nodes = len(node_indices)
print(f"Number of nodes: {num_nodes}")

# Function to process and save a single location's data
def process_and_save_location(location, location_data, output_dir):
    if location in location_to_node:
        node_id = location_to_node[location]
        lat = node_to_attrs[node_id]['latitude']
        lon = node_to_attrs[node_id]['longitude']

        # Vectorized feature creation
        features = location_data[['day_of_year', 'month', 'mean_2m_air_temperature', 'relative_humidity', 'total_precipitation']].values
        features = np.hstack([np.array([[lat, lon]] * len(features)), features])  # Add lat/lon to each row

        # Vectorized target creation
        targets = location_data[['mean_2m_air_temperature', 'relative_humidity', 'total_precipitation']].values

        # Save the processed data to disk
        output_path = os.path.join(output_dir, f"{location}.npz")
        np.savez(output_path, features=features, targets=targets)

    return location  # Return the location name for tracking

# Create a directory to save processed data
output_dir = r"C:\Users\LENOVO\semproject_trainer\processed_data"
os.makedirs(output_dir, exist_ok=True)

# Group weather data by location
print("Grouping weather data by location...")
grouped_weather = weather_df.groupby('Location_x')

# Process each location's data in parallel and save to disk
print("Processing location data...")
results = Parallel(n_jobs=-1)(
    delayed(process_and_save_location)(location, group, output_dir)
    for location, group in tqdm(grouped_weather, desc="Processing locations")
)

# Step 3: Prepare Edge Indices
edge_indices = []
print("Preparing edge indices...")
for edge in tqdm(G.edges(), desc="Processing edges"):
    if edge[0] in node_indices and edge[1] in node_indices:  # Only include edges for matched nodes
        source = node_indices[edge[0]]
        target = node_indices[edge[1]]
        edge_indices.append([source, target])
    else:
        print(f"Edge skipped: {edge} (nodes not found in node_indices)")

# Debug: Check if edge_indices is populated
print(f"Number of edges after filtering: {len(edge_indices)}")
if len(edge_indices) == 0:
    print("Warning: No edges found. Adding self-loops.")
    self_loops = torch.tensor([[i, i] for i in range(num_nodes)], dtype=torch.long).t().contiguous()
    edge_index = self_loops
else:
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

# Move edge_index to the correct device
edge_index = edge_index.to(device)

# Step 4: Define the Spatio-Temporal GNN (STGNN)
class STGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.2):
        super(STGNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Spatial layers (GCN)
        self.spatial_conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = BatchNorm(hidden_dim)
        self.spatial_conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)

        # Temporal layer (GRU)
        self.temporal_rnn = GRU(hidden_dim, hidden_dim, batch_first=True)

        # Final prediction layer
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

        # Dropout for regularization
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x, edge_index, batch_size, sequence_length):
        # Debug: Print input tensor shape
        print(f"Input tensor shape: {x.shape}")

        # Reshape input for temporal modeling: (batch_size, sequence_length, num_nodes, input_dim)
        if x.dim() == 3:  # If the input tensor is missing the num_nodes dimension
            x = x.unsqueeze(2).expand(-1, -1, num_nodes, -1)  # Expand to include num_nodes dimension

        if x.size(1) != sequence_length or x.size(2) != num_nodes or x.size(3) != self.input_dim:
            raise ValueError(f"Input tensor shape {x.shape} does not match expected shape [batch_size, sequence_length, num_nodes, input_dim]")

        # Create a mask tensor (1 for real data, 0 for padded data)
        mask = (x != 0).any(dim=-1).float()

        # Process each time step
        temporal_outputs = []
        for t in range(sequence_length):
            # Spatial aggregation for time step t
            x_t = x[:, t, :, :]  # (batch_size, num_nodes, input_dim)
            print(f"Time step {t} tensor shape: {x_t.shape}")

            # Use .reshape() instead of .view() to handle non-contiguous tensors
            x_t = x_t.reshape(-1, x_t.size(2))  # (batch_size * num_nodes, input_dim)
            print(f"Flattened time step {t} tensor shape: {x_t.shape}")

            # First spatial layer
            x_t = self.spatial_conv1(x_t, edge_index)
            x_t = self.bn1(x_t)
            x_t = F.relu(x_t)
            x_t = self.dropout(x_t)

            # Second spatial layer
            x_t = self.spatial_conv2(x_t, edge_index)
            x_t = self.bn2(x_t)
            x_t = F.relu(x_t)
            x_t = self.dropout(x_t)

            # Reshape back to (batch_size, num_nodes, hidden_dim)
            x_t = x_t.reshape(batch_size, -1, x_t.size(1))
            print(f"Reshaped back time step {t} tensor shape: {x_t.shape}")
            temporal_outputs.append(x_t)

        # Stack temporal outputs: (batch_size, sequence_length, num_nodes, hidden_dim)
        temporal_outputs = torch.stack(temporal_outputs, dim=1)
        print(f"Stacked temporal outputs shape: {temporal_outputs.shape}")

        # Reshape for GRU: (batch_size * num_nodes, sequence_length, hidden_dim)
        temporal_outputs = temporal_outputs.permute(0, 2, 1, 3).contiguous()
        temporal_outputs = temporal_outputs.reshape(-1, sequence_length, temporal_outputs.size(3))
        print(f"Reshaped for GRU tensor shape: {temporal_outputs.shape}")

        # Temporal modeling with GRU
        _, h_n = self.temporal_rnn(temporal_outputs)  # h_n: (1, batch_size * num_nodes, hidden_dim)
        h_n = h_n.squeeze(0)  # (batch_size * num_nodes, hidden_dim)
        print(f"GRU output shape: {h_n.shape}")

        # Final prediction layer
        predictions = self.fc(h_n)  # (batch_size * num_nodes, output_dim)
        print(f"Final predictions shape: {predictions.shape}")

        return predictions, mask

# Define sequence length and batch size (reduced to save memory)
sequence_length = 20  # Reduced sequence length
batch_size = 16  # Reduced batch size

# Custom Dataset for lazy loading
class WeatherDataset(Dataset):
    def __init__(self, file_paths, sequence_length, num_nodes, input_dim):
        self.file_paths = file_paths
        self.sequence_length = sequence_length
        self.num_nodes = num_nodes
        self.input_dim = input_dim

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        data = np.load(file_path)
        features = torch.tensor(data['features'], dtype=torch.float)
        targets = torch.tensor(data['targets'], dtype=torch.float)

        # Create sequences for this file
        sequences = []
        for i in range(len(features) - self.sequence_length):
            sequence_features = features[i:i + self.sequence_length]  # (sequence_length, input_dim)
            sequence_features = sequence_features.unsqueeze(1).expand(-1, self.num_nodes, -1)  # (sequence_length, num_nodes, input_dim)
            sequences.append((sequence_features, targets[i + self.sequence_length]))

        return sequences

# Get list of processed location files
location_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.npz')]

# Initialize the STGNN model
input_dim = 7  # Number of features per node (lat, lon, day_of_year, month, temp, humidity, precipitation)
hidden_dim = 32  # Hidden dimension for GNN and GRU layers
output_dim = 3  # Output dimension (temperature, humidity, precipitation)
dropout_rate = 0.2  # Adjusted dropout rate for better regularization

model = STGNN(input_dim, hidden_dim, output_dim, dropout_rate).to(device)

# Initialize optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
loss_fn = torch.nn.MSELoss()

# Checkpoint directory
checkpoint_dir = r"C:\Users\LENOVO\Desktop\semproject_trainer\checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# Training loop
print("Training the model...")
max_batches = 5  # Stop after 5 batches for debugging
for epoch in tqdm(range(100), desc="Epochs"):
    model.train()
    epoch_loss = 0.0

    # Iterate through each file
    for file_path in tqdm(location_files, desc=f"Epoch {epoch + 1}", leave=False):
        # Load sequences from the current file
        data = np.load(file_path)
        features = torch.tensor(data['features'], dtype=torch.float)
        targets = torch.tensor(data['targets'], dtype=torch.float)

        # Create sequences for this file
        sequences = []
        for i in range(len(features) - sequence_length):
            sequence_features = features[i:i + sequence_length]  # (sequence_length, input_dim)
            sequence_features = sequence_features.unsqueeze(1).expand(-1, num_nodes, -1)  # (sequence_length, num_nodes, input_dim)
            sequences.append((sequence_features, targets[i + sequence_length]))

        # Create a DataLoader for the current file
        dataset = TensorDataset(torch.stack([s[0] for s in sequences]), torch.stack([s[1] for s in sequences]))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Iterate through the dataloader
        for batch_idx, (batch_sequences, batch_targets) in enumerate(dataloader):
            if batch_idx >= max_batches:
                break  # Stop after max_batches for debugging
            print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(dataloader)}")

            # Move data to the device (GPU if available)
            batch_sequences = batch_sequences.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)

            # Forward pass
            optimizer.zero_grad()
            out, mask = model(batch_sequences, edge_index, batch_size, sequence_length)

            # Reshape out to match batch_targets shape
            out = out.view(batch_size, num_nodes, -1)  # Reshape to [batch_size, num_nodes, output_dim]

            # Fix the shape of mask
            if mask.dim() == 3:  # If mask has 3 dimensions [batch_size, sequence_length, num_nodes]
                mask = mask[:, 0, :]  # Reduce to [batch_size, num_nodes]
            elif mask.dim() == 4:  # If mask has 4 dimensions [batch_size, sequence_length, num_nodes, 1]
                mask = mask[:, 0, :, 0]  # Reduce to [batch_size, num_nodes]
            else:
                raise ValueError(f"Unexpected mask shape: {mask.shape}")

            # Expand mask to match out and batch_targets shape
            mask = mask.unsqueeze(-1).expand(-1, -1, out.size(-1))  # [batch_size, num_nodes, output_dim]

            # Expand batch_targets to match out and mask shape
            batch_targets = batch_targets.unsqueeze(1).expand(-1, num_nodes, -1)  # [batch_size, num_nodes, output_dim]

            # Debug: Print shapes once per batch
            if batch_idx == 0:
                print(f"Batch sequences shape: {batch_sequences.shape}")
                print(f"out shape: {out.shape}")
                print(f"batch_targets shape: {batch_targets.shape}")
                print(f"mask shape: {mask.shape}")

            # Compute loss (ignore padded values using the mask)
            loss = loss_fn(out * mask, batch_targets * mask)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

    # Average loss for the epoch
    epoch_loss /= len(location_files)  # Adjust based on the number of files
    scheduler.step(epoch_loss)  # Adjust learning rate based on epoch loss

    print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

    # Save checkpoint every 5 epochs
    if (epoch + 1) % 5 == 0:
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
        }, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")