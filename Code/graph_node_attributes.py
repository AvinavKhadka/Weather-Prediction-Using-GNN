import networkx as nx

# Load the graph
graph_file_path = r"C:\Users\LENOVO\Desktop\semproject\weather_graph_with_deltas.gexf"
G = nx.read_gexf(graph_file_path)

# Extract location attributes
locations = []
for node, attrs in G.nodes(data=True):
    if 'location' in attrs:
        locations.append(attrs['location'].strip())  # Strip extra spaces
    else:
        print(f"Warning: Node '{node}' has no 'location' attribute.")

# Print locations to the console
print(f"Number of nodes with location attributes: {len(locations)}")
print("\nLocations:")
for location in locations:
    print(location)

# Save locations to a text file
output_file = "graph_locations.txt"
with open(output_file, "w") as f:
    for location in locations:
        f.write(f"{location}\n")

print(f"\nLocations saved to '{output_file}'.")