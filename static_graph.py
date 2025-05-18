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

# Manually set static layout positions
pos = {
    "Collision": (0, 2),
    "Front": (0, 1),
    "FrontLeft": (-1, 1),
    "Left": (-2, 1),
    "Right": (2, 1),
    "FrontRight": (1, 1),
}

def update_graph(root_nodes, risk_flag=None):
    """Update dynamic root nodes and draw updated graph with collision risk."""
    # Remove old dynamic nodes
    dynamic_nodes = set(G.nodes()) - set(position_nodes) - {collision_node}
    G.remove_nodes_from(dynamic_nodes)

    y_bottom = 0  # Root nodes level
    x_offset = -len(root_nodes) / 2  # Spread root nodes horizontally

    for i, (root, position) in enumerate(root_nodes.items()):
        G.add_node(root)
        G.add_edge(root, position)

        # Set object node position
        pos[root] = (x_offset + i, y_bottom)

    # Color assignment
    final_colors = []
    is_high_risk = (risk_flag.value == 1) if risk_flag else False

    for node in G.nodes():
        if node == "Collision":
            final_colors.append("red" if is_high_risk else "green")
        elif node in position_nodes:
            final_colors.append("lightgray")
        else:
            final_colors.append("skyblue")

    # Draw updated graph
    plt.clf()
    nx.draw(
        G, pos,
        with_labels=True,
        node_size=2500,
        node_color=final_colors,
        font_size=10,
        font_weight="bold",
        arrows=True
    )

    # ✅ Axis settings and risk message
    plt.xlim(-2.5, 2.5)
    plt.ylim(-0.3, 2.5)
    #plt.axis("off")

    message = (
        "!!! Potential Chances of Collision: Slow down or stop !!!"
        if is_high_risk else
        "No potential risk: Continue Normal Movement"
    )

    plt.text(0, 2.3, message, fontsize=12, ha='center', color="red" if is_high_risk else "green")
    plt.title("Risk Assessment Graph", fontsize=14)
    plt.draw()
    plt.pause(0.001)
