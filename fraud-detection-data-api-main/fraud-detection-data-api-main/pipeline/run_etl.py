"""
run_etl.py

Automated ETL script for the fraud detection pipeline.
Extracts transaction data, transforms it into a graph, and loads it for model use.
"""
from pathlib import Path
import pandas as pd
from pipeline.data_pipeline import DataPipeline

def main():
    # Step 1: Extract
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_PATH = BASE_DIR / "data" 

    pipeline = DataPipeline(data_dir=DATA_PATH)
    
    df = pipeline.load_transaction_data("transactions.csv")
    print("[Extract] Loaded data:")
    print(df.head())

    # Step 2: Transform
    G = pipeline.build_heterogeneous_graph(df)
    print(f"[Transform] Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    hetero_data = pipeline.to_pyg_heterodata()
    print("[Transform] Converted to PyTorch Geometric HeteroData.")

    # Step 3: Load (for model)
    # Here, you would pass hetero_data to your model for training or inference
    # For demo, just print summary
    print("[Load] HeteroData summary:")
    print(hetero_data)

    # Step 4: Visualize (集中展示)
    print("[Visualize] Displaying transaction graph...")
    pipeline.visualize_graph()

if __name__ == "__main__":
    main()
