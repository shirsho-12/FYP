import torch
import dgl
import networkx as nx


class HGGrapth:
    def __init__(self, graph):
        self.graph = graph
        self.hg = nx.Graph()
        self.hg.add_nodes_from(graph.nodes(data=True))
        self.hg.add_edges_from(graph.edges(data=True))
        self.hg = dgl.from_networkx(self.hg)
        self.hg = dgl.add_self_loop(self.hg)
        self.hg.ndata["feat"] = self.hg.ndata["feat"].float()
        self.hg.edata["feat"] = self.hg.edata["feat"].float()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hg = self.hg.to(self.device)

    def get_graph(self):
        return self.hg

    def get_device(self):
        return self.device

    def get_graph_data(self):
        return self.hg.ndata["feat"], self.hg.edata["feat"]

    def get_graph_nodes(self):
        return self.hg.nodes()

    def get_graph_edges(self):
        return self.hg.edges()

    def get_graph_node_features(self):
        return self.hg.ndata["feat"]

    def get_graph_edge_features(self):
        return self.hg.edata["feat"]

    def get_graph_node_feature(self, node):
        return self.hg.nodes[node]

    def get_graph_edge_feature(self, edge):
        return self.hg.edges[edge]
