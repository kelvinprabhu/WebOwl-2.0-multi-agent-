# OfflineKnowledgeRetriever.py -- only when connection error with neo4j data

import os
import pickle
import json
import faiss
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import sqlite3
from dataclasses import asdict

class OfflineKnowledgeRetriever:
    """Offline version of KnowledgeRetriever that doesn't need Neo4j"""
    
    def __init__(self, index_path: str = None):
        self.embedder = None
        self.faiss_index = None
        self.chunk_metadata = {}  # Store chunk data locally
        self.index_to_chunk_id = {}
        self.chunk_id_to_index = {}
        self.index_path = index_path or "retriever_offline"
        
    def load_from_online_retriever(self, online_retriever):
        """Load data from an online retriever instance"""
        self.embedder = online_retriever.embedder
        self.faiss_index = online_retriever.faiss_index
        self.index_to_chunk_id = online_retriever.index_to_chunk_id.copy()
        self.chunk_id_to_index = online_retriever.chunk_id_to_index.copy()
        
        # Export chunk metadata from Neo4j
        print("Exporting chunk metadata from Neo4j...")
        with online_retriever.driver.session() as session:
            result = session.run("""
                MATCH (c:Chunk)
                OPTIONAL MATCH (p:Page)-[:HAS_CHUNK]->(c)
                OPTIONAL MATCH (a:Asset)-[:HAS_CHUNK]->(c)
                RETURN c.id as chunk_id, c.text as text, c.modality as modality,
                       p.url as page_url, a.url as asset_url, a.filename as asset_filename, 
                       a.type as asset_type
            """)
            
            for row in result:
                chunk_id = row['chunk_id']
                self.chunk_metadata[chunk_id] = {
                    'text': row['text'],
                    'modality': row['modality'],
                    'page_url': row['page_url'],
                    'asset_url': row['asset_url'],
                    'asset_filename': row['asset_filename'],
                    'asset_type': row['asset_type']
                }
        
        print(f"Loaded metadata for {len(self.chunk_metadata)} chunks")
        
    def save_offline(self, base_path: str = None):
        """Save all retriever data for offline use"""
        if base_path:
            self.index_path = base_path
            
        # Create directory
        Path(self.index_path).mkdir(parents=True, exist_ok=True)
        
        print(f"Saving retriever to {self.index_path}...")
        
        # 1. Save FAISS index
        if self.faiss_index:
            faiss.write_index(self.faiss_index, f"{self.index_path}/faiss_index.bin")
            print("✓ FAISS index saved")
        
        # 2. Save mappings
        with open(f"{self.index_path}/mappings.json", 'w') as f:
            json.dump({
                'index_to_chunk_id': self.index_to_chunk_id,
                'chunk_id_to_index': self.chunk_id_to_index
            }, f, indent=2)
        print("✓ Index mappings saved")
        
        # 3. Save chunk metadata to SQLite for efficient querying
        conn = sqlite3.connect(f"{self.index_path}/chunks.db")
        cursor = conn.cursor()
        
        # Create table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                text TEXT,
                modality TEXT,
                page_url TEXT,
                asset_url TEXT,
                asset_filename TEXT,
                asset_type TEXT,
                source_url TEXT,
                source_type TEXT,
                source_title TEXT
            )
        """)
        
        # Insert data
        for chunk_id, data in self.chunk_metadata.items():
            source_url = data['page_url'] or data['asset_url']
            source_type = 'page' if data['page_url'] else 'asset'
            source_title = data['page_url'] or data['asset_filename']
            
            cursor.execute("""
                INSERT OR REPLACE INTO chunks 
                (chunk_id, text, modality, page_url, asset_url, asset_filename, 
                 asset_type, source_url, source_type, source_title)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                chunk_id, data['text'], data['modality'], data['page_url'],
                data['asset_url'], data['asset_filename'], data['asset_type'],
                source_url, source_type, source_title
            ))
        
        conn.commit()
        conn.close()
        print("✓ Chunk metadata saved to SQLite")
        
        # 4. Save embedder info
        with open(f"{self.index_path}/config.json", 'w') as f:
            json.dump({
                'model_name': self.embedder.get_sentence_embedding_dimension() if hasattr(self.embedder, 'get_sentence_embedding_dimension') else 'unknown',
                'total_chunks': len(self.chunk_metadata),
                'index_size': len(self.index_to_chunk_id)
            }, f, indent=2)
        print("✓ Configuration saved")
        
        print(f"✅ Offline retriever saved successfully to {self.index_path}")
        return self.index_path
        
    def load_offline(self, index_path: str = None):
        """Load retriever from offline files"""
        if index_path:
            self.index_path = index_path
            
        if not Path(self.index_path).exists():
            raise FileNotFoundError(f"Offline index not found at {self.index_path}")
            
        print(f"Loading offline retriever from {self.index_path}...")
        
        # 1. Load embedder (you'll need to specify the model)
        from sentence_transformers import SentenceTransformer
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")  # Default model
        print("✓ Embedder loaded")
        
        # 2. Load FAISS index
        faiss_path = f"{self.index_path}/faiss_index.bin"
        if os.path.exists(faiss_path):
            self.faiss_index = faiss.read_index(faiss_path)
            print("✓ FAISS index loaded")
        
        # 3. Load mappings
        with open(f"{self.index_path}/mappings.json", 'r') as f:
            mappings = json.load(f)
            # Convert string keys back to integers for index_to_chunk_id
            self.index_to_chunk_id = {int(k): v for k, v in mappings['index_to_chunk_id'].items()}
            self.chunk_id_to_index = mappings['chunk_id_to_index']
        print("✓ Mappings loaded")
        
        # 4. Load config
        with open(f"{self.index_path}/config.json", 'r') as f:
            config = json.load(f)
        print(f"✓ Configuration loaded: {config['total_chunks']} chunks")
        
        print("✅ Offline retriever loaded successfully")
        return True
        
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search using offline index"""
        if not self.faiss_index or not self.embedder:
            raise ValueError("Retriever not properly loaded. Call load_offline() first.")
            
        # Embed query
        query_embedding = self.embedder.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search FAISS index
        scores, indices = self.faiss_index.search(query_embedding.astype(np.float32), top_k)
        
        # Get chunk details from SQLite
        conn = sqlite3.connect(f"{self.index_path}/chunks.db")
        cursor = conn.cursor()
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx in self.index_to_chunk_id:
                chunk_id = self.index_to_chunk_id[idx]
                
                cursor.execute("""
                    SELECT * FROM chunks WHERE chunk_id = ?
                """, (chunk_id,))
                
                row = cursor.fetchone()
                if row:
                    results.append({
                        'chunk_id': row[0],
                        'text': row[1],
                        'modality': row[2],
                        'source_url': row[7],
                        'source_type': row[8],
                        'source_title': row[9],
                        'score': float(scores[0][i])
                    })
        
        conn.close()
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the offline index"""
        conn = sqlite3.connect(f"{self.index_path}/chunks.db")
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM chunks")
        total_chunks = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT source_url) FROM chunks")
        unique_sources = cursor.fetchone()[0]
        
        cursor.execute("SELECT source_type, COUNT(*) FROM chunks GROUP BY source_type")
        type_counts = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            'total_chunks': total_chunks,
            'unique_sources': unique_sources,
            'type_distribution': type_counts,
            'index_size': self.faiss_index.ntotal if self.faiss_index else 0,
            'index_path': self.index_path
        }

# Helper functions for easy export/import
def export_retriever_offline(online_retriever, save_path: str = "retriever_offline"):
    """Export an online retriever to offline format"""
    offline_retriever = OfflineKnowledgeRetriever()
    offline_retriever.load_from_online_retriever(online_retriever)
    return offline_retriever.save_offline(save_path)

def load_retriever_offline(index_path: str = "retriever_offline"):
    """Load an offline retriever"""
    offline_retriever = OfflineKnowledgeRetriever()
    offline_retriever.load_offline(index_path)
    return offline_retriever

# Usage example
def example_usage():
    """Example of how to save and load offline retriever"""
    
    # Assume you have your online retriever working
    # from neo4j import GraphDatabase
    # from knowledge_retriever import KnowledgeRetriever
    
    # # 1. Export from online to offline
    # driver = GraphDatabase.driver("neo4j+s://your-uri", auth=("neo4j", "password"))
    # online_retriever = KnowledgeRetriever(driver)
    # online_retriever.build_vector_index()
    
    # Save offline
    offline_path = export_retriever_offline(online_retriever, "my_retriever_offline")
    print(f"Saved to: {offline_path}")
    
    # 2. Load and use offline (no Neo4j needed)
    offline_retriever = load_retriever_offline("my_retriever_offline")
    
    # Search offline
    results = offline_retriever.search("master programs", top_k=5)
    
    for i, result in enumerate(results, 1):
        print(f"{i}. Score: {result['score']:.3f}")
        print(f"   Source: {result['source_url']}")
        print(f"   Text: {result['text'][:100]}...")
        print()
    
    # Get stats
    stats = offline_retriever.get_stats()
    print("Index Statistics:")
    print(json.dumps(stats, indent=2))

