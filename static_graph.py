import networkx as nx
import matplotlib.pyplot as plt

plt.ion()  # Enable interactive mode

# Graph definition
position_nodes = ["Front", "Right", "Left", "Back"]
collision_node = "Collision"

G = nx.DiGraph()

def update_graph(root_nodes):
    """ Update the graph dynamically with detected root nodes. """
    G.clear()

    # Connect position nodes to Collision
    G.add_edges_from([
        ("Front", "Collision"),
        ("Right", "Collision"),
        ("Left", "Collision"),
        ("Back", "Collision")
    ])

    # Connect each object (root node) to a position
    for root, position in root_nodes.items():
        G.add_edge(root, position)

    # Plot graph
    plt.clf()
    pos = nx.spring_layout(G, seed=42)
    nx.draw(
        G, pos,
        with_labels=True,
        node_size=2000,
        node_color="lightblue",
        font_size=10,
        font_weight="bold"
    )
    plt.title("Dynamic Graph for Robot's Risk Assessment")
    plt.draw()
    plt.pause(0.001)
