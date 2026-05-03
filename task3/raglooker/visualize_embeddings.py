import chromadb
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import random

def visualize():
    base_dir = Path(__file__).resolve().parent
    chroma_path = base_dir / ".chroma_db"
    
    if not chroma_path.exists():
        print("Error: .chroma_db directory does not exist yet. Please run the app first.")
        return

    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_collection(name="steam_games")
    
    # Fetch a sample of games (plotting 5000 might be too slow/cluttered)
    sample_size = 1000
    count = collection.count()
    if count == 0:
        print("No games indexed yet.")
        return
        
    print(f"Fetching {min(sample_size, count)} games from the database...")
    try:
        # First try to get everything
        results = collection.get(include=['embeddings', 'metadatas', 'documents'], limit=sample_size)
    except Exception as e:
        print(f"Warning: Failed to fetch embeddings directly: {e}")
        print("Attempting to fetch without embeddings to verify data...")
        results = collection.get(include=['metadatas', 'documents'], limit=sample_size)
        if results['ids']:
            print(f"Successfully fetched {len(results['ids'])} games, but embeddings are missing or inaccessible.")
            print("Tip: You might need to delete the '.chroma_db' folder and restart the app to re-generate correct embeddings.")
        return
    
    embeddings = np.array(results['embeddings'])
    metadatas = results['metadatas']
    names = [m['name'] for m in metadatas]
    
    # Dimensionality Reduction (PCA) to 2D
    print("Reducing dimensions using PCA...")
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    # Plotting
    plt.figure(figsize=(12, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5, c='blue', edgecolors='white', s=30)
    
    # Annotate a few random points
    num_annotations = 15
    indices = random.sample(range(len(names)), min(num_annotations, len(names)))
    for i in indices:
        plt.annotate(names[i], (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=8, alpha=0.7)
    
    plt.title(f"Visualizing Steam Game Embeddings (PCA) - {len(names)} games")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Save the plot
    output_path = base_dir / "embeddings_visualization.png"
    plt.savefig(output_path)
    print(f"Success! Visualization saved to: {output_path}")
    plt.show()

if __name__ == "__main__":
    visualize()
