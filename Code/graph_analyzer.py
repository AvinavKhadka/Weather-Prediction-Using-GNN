import networkx as nx
import matplotlib.pyplot as plt

# Load the graph from the .gexf file
file_path = "weather_graph_with_deltas.gexf"
G = nx.read_gexf(file_path)

# Step 1: Check for edges
if G.number_of_edges() == 0:
    print("The graph has no edges.")
else:
    print(f"The graph has {G.number_of_edges()} edges.")

# Step 2: Analyze the graph
print("\nGraph Analysis:")
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")
print(f"Is the graph connected? {nx.is_connected(G)}")

# Degree distribution
degrees = [G.degree(node) for node in G.nodes()]
print(f"Average degree: {sum(degrees) / len(degrees):.2f}")
print(f"Maximum degree: {max(degrees)}")
print(f"Minimum degree: {min(degrees)}")

# Step 3: Visualize the graph
print("\nVisualizing the graph...")

# Create a layout for the graph
pos = nx.spring_layout(G, seed=42)  # Use a fixed seed for reproducibility

# Draw nodes and edges
plt.figure(figsize=(10, 8))
nx.draw_networkx_nodes(G, pos, node_size=50, node_color="blue", alpha=0.8)
nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=0.5)

# Draw labels (optional, can be slow for large graphs)
# nx.draw_networkx_labels(G, pos, font_size=8, font_color="black")

plt.title("Weather Graph Visualization")
plt.axis("off")  # Turn off the axis
plt.show()