import chromadb
import os
from pathlib import Path
import subprocess

def get_dir_size(path):
    try:
        output = subprocess.check_output(['du', '-sh', str(path)]).split()[0].decode('utf-8')
        return output
    except:
        return "Unknown"

def inspect():
    base_dir = Path(__file__).resolve().parent
    chroma_path = base_dir / ".chroma_db"
    
    print(f"--- ChromaDB Inspection ---")
    print(f"Database Location: {chroma_path}")
    
    if not chroma_path.exists():
        print("Error: .chroma_db directory does not exist yet. Please run the app first.")
        return

    print(f"Total Disk Size: {get_dir_size(chroma_path)}")
    
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_collection(name="steam_games")
    
    count = collection.count()
    print(f"Total Indexed Games: {count}")
    
    if count > 0:
        print("\n--- Sample Entries (First 5) ---")
        results = collection.get(limit=5)
        for i in range(len(results['ids'])):
            print(f"ID: {results['ids'][i]}")
            print(f"Name: {results['metadatas'][i]['name']}")
            print(f"Document Snippet: {results['documents'][i][:100]}...")
            print("-" * 30)

if __name__ == "__main__":
    inspect()
