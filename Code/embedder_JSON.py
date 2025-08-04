import torch
import networkx as nx
import pandas as pd
from tqdm import tqdm
from sklearn.neighbors import BallTree
from math import radians

# Ensure CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your merged weather and location data
data_path = r"C:\Users\LENOVO\Desktop\semproject\deltas_weather_data_cleaned.csv"
weather_df = pd.read_csv(data_path)

def create_weather_graph(weather_df, distance_threshold=100):
    """
    Create a weather graph using BallTree for efficient Haversine distance computation.
    Nodes are added only for unique locations.
    """
    try:
        G = nx.Graph()

        # Extract unique locations (latitude, longitude) for each station
        unique_locations = weather_df[['Location_x', 'Latitude', 'Longitude']].drop_duplicates()
        coordinates = unique_locations[['Latitude', 'Longitude']].values
        coordinates_rad = [[radians(lat), radians(lon)] for lat, lon in coordinates]

        # Build BallTree for nearest-neighbor search
        tree = BallTree(coordinates_rad, metric='haversine')

        # Add nodes to the graph with latitude, longitude, and location
        for _, row in tqdm(unique_locations.iterrows(), total=unique_locations.shape[0], desc="Adding Nodes"):
            station = row['Location_x']
            lat = row['Latitude']
            lon = row['Longitude']

            # Add the node with latitude, longitude, and location
            G.add_node(
                station,
                latitude=lat,
                longitude=lon,
                location=station
            )

            # Print the node coordinates only once when the node is first created
            print(f"Node {station} created with coordinates: Latitude={lat}, Longitude={lon}")

        # Add edges based on distance
        distance_threshold_rad = distance_threshold / 6371.0  # Convert km to radians

        for idx, coord in tqdm(enumerate(coordinates_rad), total=len(coordinates_rad), desc="Adding Edges"):
            indices = tree.query_radius([coord], r=distance_threshold_rad)[0]

            for neighbor_idx in indices:
                neighbor_idx = int(neighbor_idx)
                if neighbor_idx > idx:  # Avoid duplicate edges
                    station_i = unique_locations.iloc[idx]['Location_x']
                    station_j = unique_locations.iloc[neighbor_idx]['Location_x']
                    dist = haversine_distance(
                        coordinates[idx], coordinates[neighbor_idx]
                    )
                    G.add_edge(station_i, station_j, weight=dist)

        print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
        return G

    except Exception as e:
        print(f"Error creating weather graph: {e}")
        return None

def haversine_distance(coord1, coord2):
    """
    Compute the Haversine distance between two coordinates.
    """
    lat1, lon1 = map(radians, coord1)
    lat2, lon2 = map(radians, coord2)
    
    # Convert latitudes and longitudes to tensors
    lat1, lon1, lat2, lon2 = map(torch.tensor, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (torch.sin(dlat / 2) ** 2 +
         torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2)
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    return (c * 6371.0).item()  # Distance in kilometers

def save_graph(graph, file_path="weather_graph_with_deltas.gexf"):
    """
    Save the graph to a file in GEXF format.
    """
    try:
        nx.write_gexf(graph, file_path)
        print(f"Graph saved to {file_path}")

    except Exception as e:
        print(f"Error saving graph: {e}")

def main():
    """
    Main function to preprocess data, create a weather graph, and save it.
    """
    try:
        # Create the weather graph
        graph = create_weather_graph(weather_df, distance_threshold=100)
        if graph is not None:
            save_graph(graph, file_path="weather_graph_with_deltas.gexf")
            print("Graph creation and saving successful.")
        else:
            print("Graph creation failed.")

    except Exception as e:
        print(f"Error in the main pipeline: {e}")

# Entry point for the script
if __name__ == "__main__":
    main()