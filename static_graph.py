import networkx as nx
import matplotlib.pyplot as plt

plt.ion()  # Enable interactive mode

# Static layers
position_nodes = ["Left", "FrontLeft", "Front", "FrontRight", "Right"]
collision_node = "Collision"

# Create directed graph
G = nx.DiGraph()

# Add static nodes and edges
G.add_nodes_from(position_nodes + [collision_node])
G.add_edges_from([(pos, collision_node) for pos in position_nodes])

# Manually set static layout positions for a tree-like structure (inverted)
pos = {
    "Collision": (0, 2),
    "Front": (0, 1),
    "FrontLeft": (-1, 1),
    "Left": (-2, 1),
    "Right": (2, 1),
    "FrontRight": (1, 1),
}

def update_graph(root_nodes):
    """Update dynamic root nodes below the static structure in a tree layout."""
    # Remove old dynamic nodes
    dynamic_nodes = set(G.nodes()) - set(position_nodes) - {collision_node}
    G.remove_nodes_from(dynamic_nodes)

    # Add new root nodes and edges to position nodes
    y_bottom = 0  # Root nodes level
    x_offset = -len(root_nodes) / 2  # Spread root nodes horizontally
    for i, (root, position) in enumerate(root_nodes.items()):
        G.add_node(root)
        G.add_edge(root, position)

        # Set position just below the corresponding position node
        parent_pos = pos.get(position, (0, 1))
        pos[root] = (x_offset + i, y_bottom)

    # Draw updated graph
    plt.clf()
    nx.draw(
        G, pos,
        with_labels=True,
        node_size=2500,
        node_color="skyblue",
        font_size=10,
        font_weight="bold",
        arrows=True
    )
    plt.title("Risk Assessment Graph", fontsize=14)
    plt.draw()
    plt.pause(0.001)
