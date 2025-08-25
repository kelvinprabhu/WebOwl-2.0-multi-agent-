# KnowledgeRetriever.py

import uuid
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
from neo4j import GraphDatabase
import json
import re
from collections import defaultdict

class SearchMode(Enum):
    SEMANTIC = "semantic"
    GRAPH_WALK = "graph_walk"
    HYBRID = "hybrid"
    MULTIMODAL = "multimodal"

@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    modality: str
    score: float
    source_url: str
    source_type: str  # 'page' or 'asset'
    source_title: Optional[str] = None
    context_path: Optional[List[str]] = None  # breadcrumb of how we got here
    related_assets: Optional[List[Dict]] = None

class KnowledgeRetriever:
    def __init__(self, neo4j_driver, embedding_model="all-MiniLM-L6-v2"):
        self.driver = neo4j_driver
        self.embedder = SentenceTransformer(embedding_model)
        self.chunk_embeddings = {}
        self.faiss_index = None
        self.chunk_id_to_index = {}
        self.index_to_chunk_id = {}
        
    def build_vector_index(self):
        """Build FAISS index for semantic search"""
        print("Building vector index...")
        
        # Get all chunks
        with self.driver.session() as session:
            result = session.run("""
                MATCH (c:Chunk)
                RETURN c.id as chunk_id, c.text as text, c.modality as modality
            """)
            chunks = list(result)
            
        if not chunks:
            print("No chunks found!")
            return
            
        # Apply text splitting to match what you did during ingestion
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        
        # Split texts and maintain mapping
        all_split_texts = []
        split_to_chunk_mapping = []  # Maps split index to original chunk
        
        for chunk in chunks:
            if chunk['text'] and chunk['text'].strip():  # Check for valid text
                split_texts = text_splitter.split_text(chunk['text'])
                for split_text in split_texts:
                    if split_text.strip():  # Only add non-empty splits
                        all_split_texts.append(split_text)
                        split_to_chunk_mapping.append(chunk['chunk_id'])
        
        if not all_split_texts:
            print("No valid text found in chunks!")
            return
        
        print(f"Processing {len(all_split_texts)} text splits from {len(chunks)} chunks")
        
        # Generate embeddings for split texts
        embeddings = self.embedder.encode(all_split_texts, show_progress_bar=True)
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.faiss_index.add(embeddings.astype(np.float32))
        
        # Create mappings for splits
        self.chunk_id_to_index = {}
        self.index_to_chunk_id = {}
        
        for idx, chunk_id in enumerate(split_to_chunk_mapping):
            self.index_to_chunk_id[idx] = chunk_id
            # Note: chunk_id_to_index may have multiple indices for same chunk_id
            if chunk_id not in self.chunk_id_to_index:
                self.chunk_id_to_index[chunk_id] = []
            self.chunk_id_to_index[chunk_id].append(idx)
            
        print(f"Built index with {len(all_split_texts)} text splits from {len(chunks)} chunks")
        
    def semantic_search(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        """Pure semantic search using embeddings"""
        if self.faiss_index is None:
            raise ValueError("Vector index not built. Call build_vector_index() first.")
            
        # Embed query
        query_embedding = self.embedder.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.faiss_index.search(query_embedding.astype(np.float32), top_k)
        
        # Get chunk details from Neo4j
        chunk_ids = [self.index_to_chunk_id[idx] for idx in indices[0]]
        
        with self.driver.session() as session:
            result = session.run("""
                UNWIND $chunk_ids as chunk_id
                MATCH (c:Chunk {id: chunk_id})
                OPTIONAL MATCH (p:Page)-[:HAS_CHUNK]->(c)
                OPTIONAL MATCH (a:Asset)-[:HAS_CHUNK]->(c)
                RETURN c.id as chunk_id, c.text as text, c.modality as modality,
                       p.url as page_url, p.url as page_title,
                       a.url as asset_url, a.filename as asset_filename, a.type as asset_type
            """, chunk_ids=chunk_ids)
            
            chunk_data = {row['chunk_id']: row for row in result}
            
        # Build results
        results = []
        for i, chunk_id in enumerate(chunk_ids):
            if chunk_id in chunk_data:
                data = chunk_data[chunk_id]
                
                source_url = data['page_url'] or data['asset_url']
                source_type = 'page' if data['page_url'] else 'asset'
                source_title = data['page_title'] or data['asset_filename']
                
                results.append(RetrievedChunk(
                    chunk_id=chunk_id,
                    text=data['text'],
                    modality=data['modality'],
                    score=float(scores[0][i]),
                    source_url=source_url,
                    source_type=source_type,
                    source_title=source_title
                ))
                
        return results
    
    def graph_walk_search(self, query: str, start_urls: List[str] = None, max_depth: int = 2) -> List[RetrievedChunk]:
        """Graph-based search following relationships"""
        
        with self.driver.session() as session:
            if start_urls:
                # Start from specific URLs
                cypher = """
                UNWIND $start_urls as url
                MATCH (start:Page {url: url})
                CALL {
                    WITH start
                    MATCH path = (start)-[:LINKS_TO|CONTAINS*1..%d]-(end)
                    WHERE end:Page OR end:Asset
                    RETURN end, length(path) as depth, path
                }
                OPTIONAL MATCH (end)-[:HAS_CHUNK]->(c:Chunk)
                WHERE c.text CONTAINS $query_term OR end.title CONTAINS $query_term
                RETURN DISTINCT c.id as chunk_id, c.text as text, c.modality as modality,
                       end.url as source_url, end.title as source_title,
                       labels(end)[0] as source_type, depth,
                       [node in nodes(path) | node.url] as path_urls
                ORDER BY depth, c.id
                """ % max_depth
            else:
                # Global search
                cypher = """
                MATCH (c:Chunk)
                WHERE c.text CONTAINS $query_term
                OPTIONAL MATCH (p:Page)-[:HAS_CHUNK]->(c)
                OPTIONAL MATCH (a:Asset)-[:HAS_CHUNK]->(c)
                RETURN c.id as chunk_id, c.text as text, c.modality as modality,
                       coalesce(p.url, a.url) as source_url,
                       coalesce(p.url, a.filename) as source_title,
                       CASE WHEN p IS NOT NULL THEN 'Page' ELSE 'Asset' END as source_type,
                       0 as depth, [] as path_urls
                """
                
            # Simple keyword extraction for graph search
            query_terms = re.findall(r'\b\w+\b', query.lower())
            main_term = max(query_terms, key=len) if query_terms else query
            
            result = session.run(cypher, start_urls=start_urls or [], query_term=main_term)
            
            chunks = []
            for row in result:
                chunks.append(RetrievedChunk(
                    chunk_id=row['chunk_id'],
                    text=row['text'],
                    modality=row['modality'],
                    score=1.0 / (row['depth'] + 1),  # Higher score for closer nodes
                    source_url=row['source_url'],
                    source_type=row['source_type'].lower(),
                    source_title=row['source_title'],
                    context_path=row['path_urls']
                ))
                
        return chunks
    
    def multimodal_search(self, query: str, include_assets: bool = True, top_k: int = 10) -> List[RetrievedChunk]:
        """Search across different modalities with context"""
        
        # Start with semantic search
        semantic_results = self.semantic_search(query, top_k)
        
        # Enhance with related assets and context
        enhanced_results = []
        
        with self.driver.session() as session:
            for chunk in semantic_results:
                # Get related assets for page chunks
                if chunk.source_type == 'page' and include_assets:
                    related_assets_result = session.run("""
                        MATCH (p:Page {url: $page_url})-[:CONTAINS]->(a:Asset)
                        RETURN a.url as url, a.type as type, a.filename as filename
                        LIMIT 5
                    """, page_url=chunk.source_url)
                    
                    related_assets = [dict(row) for row in related_assets_result]
                    chunk.related_assets = related_assets if related_assets else None
                
                enhanced_results.append(chunk)
                
        return enhanced_results
    
    def hybrid_search(self, query: str, semantic_weight: float = 0.7, graph_weight: float = 0.3, 
                     top_k: int = 10) -> List[RetrievedChunk]:
        """Combine semantic and graph search"""
        
        # Get results from both methods
        semantic_results = self.semantic_search(query, top_k * 2)
        graph_results = self.graph_walk_search(query)
        
        # Combine and re-rank
        combined_scores = defaultdict(float)
        all_chunks = {}
        
        # Add semantic scores
        for chunk in semantic_results:
            combined_scores[chunk.chunk_id] += chunk.score * semantic_weight
            all_chunks[chunk.chunk_id] = chunk
            
        # Add graph scores
        for chunk in graph_results:
            combined_scores[chunk.chunk_id] += chunk.score * graph_weight
            if chunk.chunk_id not in all_chunks:
                all_chunks[chunk.chunk_id] = chunk
            else:
                # Merge context path if available
                if chunk.context_path and not all_chunks[chunk.chunk_id].context_path:
                    all_chunks[chunk.chunk_id].context_path = chunk.context_path
        
        # Sort by combined score
        sorted_chunks = sorted(
            [(chunk_id, score) for chunk_id, score in combined_scores.items()],
            key=lambda x: x[1], reverse=True
        )[:top_k]
        
        # Update scores and return
        results = []
        for chunk_id, score in sorted_chunks:
            chunk = all_chunks[chunk_id]
            chunk.score = score
            results.append(chunk)
            
        return results
    
    def get_context_window(self, chunk_id: str, window_size: int = 2) -> Dict[str, Any]:
        """Get surrounding context for a chunk"""
        
        with self.driver.session() as session:
            result = session.run("""
                MATCH (c:Chunk {id: $chunk_id})
                OPTIONAL MATCH (source)-[:HAS_CHUNK]->(c)
                WHERE source:Page OR source:Asset
                
                // Get related chunks from same source
                OPTIONAL MATCH (source)-[:HAS_CHUNK]->(related:Chunk)
                WHERE related <> c
                
                // Get connected pages/assets
                OPTIONAL MATCH (source)-[:LINKS_TO|CONTAINS]-(connected)
                WHERE connected:Page OR connected:Asset
                OPTIONAL MATCH (connected)-[:HAS_CHUNK]->(connected_chunk:Chunk)
                
                RETURN c, source,
                       collect(DISTINCT related)[..5] as related_chunks,
                       collect(DISTINCT {node: connected, chunks: collect(DISTINCT connected_chunk)})[..3] as connected_sources
            """, chunk_id=chunk_id)
            
            row = result.single()
            if not row:
                return {}
                
            return {
                'chunk': dict(row['c']),
                'source': dict(row['source']),
                'related_chunks': [dict(c) for c in row['related_chunks']],
                'connected_sources': row['connected_sources']
            }
    
    def search(self, query: str, mode: SearchMode = SearchMode.HYBRID, **kwargs) -> List[RetrievedChunk]:
        """Main search interface"""
        
        if mode == SearchMode.SEMANTIC:
            return self.semantic_search(query, **kwargs)
        elif mode == SearchMode.GRAPH_WALK:
            return self.graph_walk_search(query, **kwargs)
        elif mode == SearchMode.MULTIMODAL:
            return self.multimodal_search(query, **kwargs)
        elif mode == SearchMode.HYBRID:
            return self.hybrid_search(query, **kwargs)
        else:
            raise ValueError(f"Unknown search mode: {mode}")
    
    def format_for_llm(self, chunks: List[RetrievedChunk], include_context: bool = True) -> str:
        """Format retrieved chunks for LLM consumption"""
        
        formatted_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            part = f"## Source {i} (Score: {chunk.score:.3f})\n"
            part += f"**Type:** {chunk.source_type.title()}\n"
            part += f"**URL:** {chunk.source_url}\n"
            
            if chunk.source_title:
                part += f"**Title:** {chunk.source_title}\n"
                
            if chunk.modality != 'text':
                part += f"**Modality:** {chunk.modality}\n"
                
            if chunk.context_path and include_context:
                path_str = " → ".join(chunk.context_path[-3:])  # Last 3 steps
                part += f"**Path:** {path_str}\n"
                
            if chunk.related_assets and include_context:
                assets_str = ", ".join([f"{a['filename']} ({a['type']})" for a in chunk.related_assets[:3]])
                part += f"**Related Assets:** {assets_str}\n"
                
            part += f"\n**Content:**\n{chunk.text}\n\n"
            
            formatted_parts.append(part)
            
        return "\n".join(formatted_parts)

# # Example usage
# def example_usage():
#     # Initialize
#     driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
#     retriever = KnowledgeRetriever(driver)
    
#     # Build vector index (do this once)
#     retriever.build_vector_index()
    
#     # Different search modes
#     query = "machine learning algorithms"
    
#     # Semantic search
#     semantic_results = retriever.search(query, SearchMode.SEMANTIC, top_k=5)
    
#     # Graph walk search
#     graph_results = retriever.search(query, SearchMode.GRAPH_WALK, max_depth=2)
    
#     # Multimodal search (includes related assets)
#     multimodal_results = retriever.search(query, SearchMode.MULTIMODAL, include_assets=True)
    
#     # Hybrid search (recommended)
#     hybrid_results = retriever.search(query, SearchMode.HYBRID, top_k=10)
    
#     # Format for LLM
#     llm_context = retriever.format_for_llm(hybrid_results)
#     print("=== LLM CONTEXT ===")
#     print(llm_context)
    
#     # Get detailed context for a specific chunk
#     if hybrid_results:
#         context = retriever.get_context_window(hybrid_results[0].chunk_id)
#         print("\n=== DETAILED CONTEXT ===")
#         print(json.dumps(context, indent=2, default=str))

