"""
data_pipeline.py

This module defines the data pipeline for the real-time financial fraud detection system.
It includes loading raw transaction data, constructing a heterogeneous graph using NetworkX,
and preparing data for the GraphSAGE model in PyTorch Geometric.
"""

import os
import pandas as pd
import networkx as nx
from torch_geometric.data import HeteroData

class DataPipeline:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.graph = None
        self.hetero_data = None

    def visualize_graph(self, save_path=None):
        """Visualize the NetworkX graph and optionally save to file."""
        import matplotlib.pyplot as plt
        if self.graph is None:
            raise ValueError("Graph not built yet.")
        plt.figure(figsize=(6, 4))
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=800, font_size=10)
        edge_labels = nx.get_edge_attributes(self.graph, 'amount')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)
        plt.title("Transaction Graph")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def load_transaction_data(self, filename):
        """Load transaction data from a CSV file."""
        path = os.path.join(self.data_dir, filename)
        df = pd.read_csv(path)
        return df

    def build_heterogeneous_graph(self, transactions_df):
        """Construct a heterogeneous graph from transaction data using NetworkX."""
        G = nx.MultiDiGraph()
        for _, row in transactions_df.iterrows():
            src = row['src_account']
            dst = row['dst_account']
            amount = row['amount']
            timestamp = row['timestamp']
            G.add_node(src, node_type='account')
            G.add_node(dst, node_type='account')
            G.add_edge(src, dst, amount=amount, timestamp=timestamp, edge_type='transaction')
        self.graph = G
        return G

    def to_pyg_heterodata(self):
        """Convert the NetworkX graph to PyTorch Geometric HeteroData format."""
        if self.graph is None:
            raise ValueError("Graph not built yet.")
        data = HeteroData()
        # Example: Add nodes and edges (customize as needed)
        accounts = [n for n, attr in self.graph.nodes(data=True) if attr['node_type'] == 'account']
        data['account'].num_nodes = len(accounts)
        # Map node ids to indices
        node_id_map = {node: i for i, node in enumerate(accounts)}
        # Edges
        src, dst = [], []
        for u, v, attr in self.graph.edges(data=True):
            if attr.get('edge_type') == 'transaction':
                src.append(node_id_map[u])
                dst.append(node_id_map[v])
        import torch
        data['account', 'transaction', 'account'].edge_index = torch.tensor([src, dst], dtype=torch.long)
        self.hetero_data = data
        return data
