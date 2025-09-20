"""
ChromaDB vector store for mathematical RAG system.
Industry practice: Use persistent vector storage with metadata filtering.
"""
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings
from langchain.docstore.document import Document
import numpy as np
import logging
from pathlib import Path
import uuid

logger = logging.getLogger(__name__)


class MathematicalVectorStore:
    """
    ChromaDB-based vector store optimized for mathematical content.
    
    Industry practice: Persistent storage with rich metadata for filtering.
    """
    
    def __init__(self, 
                 persist_directory: str = "data/vector_store",
                 collection_name: str = "mathematical_problems"):
        """
        Initialize ChromaDB vector store.
        
        Args:
            persist_directory: Directory for persistent storage
            collection_name: Name of the collection to store embeddings
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,  # Disable telemetry
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Mathematical problems and solutions for RAG"}
        )
        
        logger.info(f"Vector store initialized: {self.collection.count()} documents in collection")
        
        # Statistics
        self.storage_stats = {
            "documents_added": 0,
            "documents_updated": 0,
            "queries_performed": 0
        }
    
    def add_documents(self, 
                 documents: List[Document], 
                 embeddings: List[np.ndarray]) -> None:
        """
        Add documents with embeddings to the vector store.
        
        Industry practice: Batch insertion with rich metadata.
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
        
        if not documents:
            logger.warning("No documents to add")
            return
        
        logger.info(f"Adding {len(documents)} documents to vector store...")
        
        # Prepare data for ChromaDB
        ids = []
        texts = []
        metadatas = []
        embedding_list = []
        
        for doc, embedding in zip(documents, embeddings):
            # Always generate a unique ID for vector storage
            # Use chunk_id if available, otherwise generate new UUID
            if 'chunk_id' in doc.metadata:
                doc_id = doc.metadata['chunk_id']
            else:
                doc_id = str(uuid.uuid4())
            
            ids.append(str(doc_id))
            
            # Store document content
            texts.append(doc.page_content)
            
            # Prepare metadata (ChromaDB requires JSON-serializable metadata)
            metadata = self._prepare_metadata(doc.metadata)
            metadatas.append(metadata)
            
            # Convert numpy array to list
            embedding_list.append(embedding.tolist())
        
        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embedding_list,
            documents=texts,
            metadatas=metadatas
        )
        
        self.storage_stats["documents_added"] += len(documents)
        logger.info(f"Successfully added {len(documents)} documents")
    
    def _prepare_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare metadata for ChromaDB storage.
        
        ChromaDB requires JSON-serializable metadata.
        """
        prepared = {}
        
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                prepared[key] = value
            elif isinstance(value, list):
                # Convert lists to comma-separated strings
                if all(isinstance(item, str) for item in value):
                    prepared[key] = ",".join(value)
                else:
                    prepared[key] = str(value)
            else:
                # Convert other types to string
                prepared[key] = str(value)
        
        return prepared
    
    def similarity_search(self, 
                         query_embedding: np.ndarray,
                         k: int = 5,
                         filter_dict: Optional[Dict[str, Any]] = None) -> List[Tuple[Document, float]]:
        """
        Perform similarity search in the vector store.
        
        Industry practice: Return documents with similarity scores.
        """
        logger.debug(f"Performing similarity search with k={k}")
        
        # Prepare query
        query_embedding_list = query_embedding.tolist()
        
        # Build where clause for filtering
        where_clause = {}
        if filter_dict:
            for key, value in filter_dict.items():
                where_clause[key] = value
        
        # Perform search
        results = self.collection.query(
            query_embeddings=[query_embedding_list],
            n_results=k,
            where=where_clause if where_clause else None
        )
        
        self.storage_stats["queries_performed"] += 1
        
        # Convert results to Document objects with scores
        documents_with_scores = []
        
        if results['documents'] and results['documents'][0]:
            for i, (doc_text, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                # Convert distance to similarity score (0-1, higher is more similar)
                similarity = 1.0 / (1.0 + distance)
                
                # Reconstruct Document
                doc = Document(
                    page_content=doc_text,
                    metadata=metadata
                )
                
                documents_with_scores.append((doc, similarity))
        
        logger.debug(f"Found {len(documents_with_scores)} similar documents")
        return documents_with_scores
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store collection."""
        collection_info = {
            "collection_name": self.collection_name,
            "document_count": self.collection.count(),
            "persist_directory": str(self.persist_directory),
            "storage_stats": self.storage_stats.copy()
        }
        
        return collection_info
    
    def delete_collection(self) -> None:
        """Delete the entire collection. Use with caution!"""
        logger.warning(f"Deleting collection: {self.collection_name}")
        self.client.delete_collection(name=self.collection_name)
        
        # Recreate empty collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Mathematical problems and solutions for RAG"}
        )
    
    def update_document(self, 
                       doc_id: str, 
                       document: Document, 
                       embedding: np.ndarray) -> None:
        """Update an existing document in the vector store."""
        logger.debug(f"Updating document: {doc_id}")
        
        self.collection.update(
            ids=[doc_id],
            embeddings=[embedding.tolist()],
            documents=[document.page_content],
            metadatas=[self._prepare_metadata(document.metadata)]
        )
        
        self.storage_stats["documents_updated"] += 1


class VectorStoreManager:
    """
    Manager class for vector store operations.
    
    Industry practice: Encapsulate complex operations in manager classes.
    """
    
    def __init__(self, 
                 vector_store: MathematicalVectorStore,
                 embedder):
        self.vector_store = vector_store
        self.embedder = embedder
    
    def index_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Index a batch of documents (generate embeddings and store).
        
        Industry practice: Single method for complete indexing pipeline.
        """
        if not documents:
            return {"status": "no_documents", "count": 0}
        
        logger.info(f"Indexing {len(documents)} documents...")
        
        # Generate embeddings
        texts = [doc.page_content for doc in documents]
        embeddings = self.embedder.generate_batch_embeddings(texts)
        
        # Store in vector database
        self.vector_store.add_documents(documents, embeddings)
        
        result = {
            "status": "success",
            "count": len(documents),
            "collection_stats": self.vector_store.get_collection_stats()
        }
        
        logger.info(f"Successfully indexed {len(documents)} documents")
        return result
    
    def search(self, 
               query: str, 
               k: int = 5,
               filters: Optional[Dict[str, Any]] = None) -> List[Tuple[Document, float]]:
        """
        Search for similar documents given a text query.
        """
        logger.debug(f"Searching for: {query[:100]}...")
        
        # Generate query embedding
        query_embedding = self.embedder.generate_embedding(query)
        
        # Perform similarity search
        results = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            k=k,
            filter_dict=filters
        )
        
        return results