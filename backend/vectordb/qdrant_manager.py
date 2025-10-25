"""
Qdrant Manager - Vector database operations for Islamic texts with LlamaIndex integration.

This module provides both low-level Qdrant API operations and LlamaIndex VectorStore integration.
Updated for Stage 4 with enhanced universal schema support.
"""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import requests

# LlamaIndex imports for VectorStore integration
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Updated imports for backend structure
from backend.core.config import Config
from backend.core.utils import ProgressTracker, setup_logging

logger = setup_logging("qdrant_manager")


class QdrantManager:
    """Manager class for Qdrant vector database operations with LlamaIndex integration."""
    
    def __init__(
        self, 
        url: str = None, 
        collection_name: str = None,
        vector_size: int = None,
    ):
        """
        Initialize QdrantManager.
        
        Args:
            url: Qdrant server URL (defaults to config)
            collection_name: Name of the collection to manage (defaults to config)
            vector_size: Dimension of vectors (defaults to config)
        """
        self.url = (url or Config.QDRANT_URL).rstrip('/')
        self.collection_name = collection_name or Config.QDRANT_COLLECTION_NAME
        self.vector_size = vector_size or Config.VECTOR_SIZE
        self.base_url = f"{self.url}/collections/{self.collection_name}"
        
        # Initialize Qdrant client for LlamaIndex integration
        # Extract host and port from URL
        url_parts = self.url.replace('http://', '').replace('https://', '').split(':')
        host = url_parts[0]
        port = int(url_parts[1]) if len(url_parts) > 1 else 6333
        
        try:
            self.client = QdrantClient(host=host, port=port)
            logger.info(f"Initialized Qdrant client: {host}:{port}")
        except Exception as e:
            logger.warning(f"Could not initialize Qdrant client: {e}")
            self.client = None
    
    def get_vector_store(self, collection_name: Optional[str] = None) -> QdrantVectorStore:
        """
        Get LlamaIndex QdrantVectorStore instance.
        
        Args:
            collection_name: Collection name (defaults to instance collection_name)
            
        Returns:
            QdrantVectorStore instance for LlamaIndex integration
        """
        coll_name = collection_name or self.collection_name
        
        if not self.client:
            raise RuntimeError("Qdrant client not initialized")
        
        vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=coll_name,
        )
        
        logger.info(f"Created LlamaIndex vector store for collection: {coll_name}")
        return vector_store
    
    def create_index(
        self, 
        collection_name: Optional[str] = None,
        embed_model = None
    ) -> VectorStoreIndex:
        """
        Create or load a LlamaIndex VectorStoreIndex.
        
        Args:
            collection_name: Collection name (defaults to instance collection_name)
            embed_model: LlamaIndex embedding model (uses Settings.embed_model if None)
            
        Returns:
            VectorStoreIndex instance
        """
        from llama_index.core import Settings
        
        coll_name = collection_name or self.collection_name
        embed_model = embed_model or Settings.embed_model
        
        # Ensure collection exists
        if not self.collection_exists(coll_name):
            logger.info(f"Collection '{coll_name}' not found, creating it...")
            self.create_collection(collection_name=coll_name)
        
        # Create vector store
        vector_store = self.get_vector_store(coll_name)
        
        # Create storage context
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Create or load index
        try:
            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                embed_model=embed_model,
            )
            logger.info(f"Loaded existing index for collection: {coll_name}")
        except Exception as e:
            logger.warning(f"Could not load existing index: {e}")
            index = VectorStoreIndex(
                nodes=[],
                storage_context=storage_context,
                embed_model=embed_model,
            )
            logger.info(f"Created new index for collection: {coll_name}")
        
        return index
    
    def collection_exists(self, collection_name: Optional[str] = None) -> bool:
        """
        Check if a collection exists.
        
        Args:
            collection_name: Collection name to check
            
        Returns:
            True if collection exists, False otherwise
        """
        coll_name = collection_name or self.collection_name
        
        if self.client:
            try:
                collections = self.client.get_collections().collections
                return any(c.name == coll_name for c in collections)
            except Exception as e:
                logger.error(f"Error checking collection existence: {e}")
        
        # Fallback to REST API
        try:
            info = self.inspect_collection(coll_name)
            return info is not None
        except:
            return False
    
    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        """Make HTTP request to Qdrant API."""
        url = f"{self.url}{endpoint}"
        headers = {"Content-Type": "application/json"}
        
        try:
            if method == "GET":
                response = requests.get(url, headers=headers)
            elif method == "POST":
                response = requests.post(url, headers=headers, json=data)
            elif method == "PUT":
                response = requests.put(url, headers=headers, json=data)
            elif method == "DELETE":
                response = requests.delete(url, headers=headers)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request failed: {str(e)}")
    
    def inspect_collections(self) -> Dict[str, Any]:
        """List all collections in the Qdrant instance."""
        return self._make_request("GET", "/collections")
    
    def inspect_collection(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get detailed information about a specific collection.
        
        Args:
            collection_name: Collection name (defaults to instance collection_name)
        """
        coll_name = collection_name or self.collection_name
        try:
            return self._make_request("GET", f"/collections/{coll_name}")
        except:
            return None
    
    def inspect_points(
        self, 
        limit: int = 10, 
        offset: Optional[str] = None,
        with_payload: bool = True,
        with_vector: bool = False
    ) -> Dict[str, Any]:
        """
        Scroll through points in the collection.
        
        Args:
            limit: Number of points to retrieve
            offset: Pagination offset (point ID)
            with_payload: Include payload data
            with_vector: Include vector embeddings
        """
        data = {
            "limit": limit,
            "with_payload": with_payload,
            "with_vector": with_vector
        }
        if offset:
            data["offset"] = offset
        
        return self._make_request("POST", f"/collections/{self.collection_name}/points/scroll", data)
    
    def get_point(self, point_id: str, with_vector: bool = True) -> Dict[str, Any]:
        """
        Get a specific point by ID.
        
        Args:
            point_id: UUID of the point
            with_vector: Include vector embeddings
        """
        return self._make_request("GET", f"/collections/{self.collection_name}/points/{point_id}?with_vector={str(with_vector).lower()}")
    
    def count_points(self, collection_name: Optional[str] = None) -> int:
        """Get total number of points in the collection."""
        coll_name = collection_name or self.collection_name
        info = self.inspect_collection(coll_name)
        if info:
            return info.get("result", {}).get("points_count", 0)
        return 0
    
    def export_collection(
        self, 
        output_file: str, 
        batch_size: int = 100,
        with_vectors: bool = True
    ) -> Dict[str, Any]:
        """
        Export entire collection to a JSON file.
        
        Args:
            output_file: Path to output JSON file
            batch_size: Number of points to fetch per batch
            with_vectors: Include vector embeddings in export
            
        Returns:
            Dictionary with export statistics
        """
        all_points = []
        offset = None
        total_fetched = 0
        
        if not self.inspect_collection():
            print(f"Collection '{self.collection_name}' not found")
            return {"points_exported": 0, "output_file": output_file, "with_vectors": with_vectors}
        
        print(f"Exporting collection '{self.collection_name}'...")

        while True:
            result = self.inspect_points(
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vector=with_vectors
            )
            
            points = result.get("result", {}).get("points", [])
            if not points:
                break
            
            all_points.extend(points)
            total_fetched += len(points)
            print(f"Fetched {total_fetched} points...")
            
            next_offset = result.get("result", {}).get("next_page_offset")
            if not next_offset or next_offset == offset:
                break
            offset = next_offset
        
        # Get collection info
        collection_info = self.inspect_collection()
        
        export_data = {
            "collection_name": self.collection_name,
            "export_timestamp": datetime.utcnow().isoformat(),
            "collection_config": collection_info.get("result", {}).get("config", {}),
            "points_count": len(all_points),
            "points": all_points
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        stats = {
            "points_exported": len(all_points),
            "output_file": output_file,
            "with_vectors": with_vectors
        }
        
        print(f"Export complete: {len(all_points)} points saved to {output_file}")
        return stats
    
    def import_collection(
        self, 
        input_file: str,
        batch_size: int = 100,
        recreate_collection: bool = False
    ) -> Dict[str, Any]:
        """
        Import points from a JSON file into the collection.
        
        Args:
            input_file: Path to input JSON file
            batch_size: Number of points to upload per batch
            recreate_collection: If True, delete and recreate collection before import
            
        Returns:
            Dictionary with import statistics
        """
        with open(input_file, 'r', encoding='utf-8') as f:
            import_data = json.load(f)
        
        points = import_data.get("points", [])
        collection_config = import_data.get("collection_config", {})
        
        print(f"Importing {len(points)} points into collection '{self.collection_name}'...")
        
        # Recreate collection if requested
        if recreate_collection:
            try:
                self.clear_collection(delete_collection=True)
                print(f"Deleted existing collection '{self.collection_name}'")
            except:
                pass
            
            # Create collection with config from export
            params = collection_config.get("params", {})
            if params:
                create_data = {
                    "vectors": params.get("vectors", {"size": self.vector_size, "distance": "Cosine"}),
                    "shard_number": params.get("shard_number", 1),
                    "replication_factor": params.get("replication_factor", 1),
                    "on_disk_payload": params.get("on_disk_payload", True)
                }
                self._make_request("PUT", f"/collections/{self.collection_name}", create_data)
                print(f"Created collection '{self.collection_name}'")
        
        # Upload points in batches
        total_uploaded = 0
        tracker = ProgressTracker(len(points), "Importing points")
        
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            
            # Format points for upsert
            formatted_points = []
            for point in batch:
                formatted_point = {
                    "id": point["id"],
                    "vector": point.get("vector", []),
                    "payload": point.get("payload", {})
                }
                formatted_points.append(formatted_point)
            
            upsert_data = {"points": formatted_points}
            self._make_request("PUT", f"/collections/{self.collection_name}/points", upsert_data)
            
            total_uploaded += len(batch)
            tracker.update(len(batch))
        
        tracker.finish()
        
        stats = {
            "points_imported": total_uploaded,
            "input_file": input_file,
            "recreate_collection": recreate_collection
        }
        
        print(f"Import complete: {total_uploaded} points imported")
        return stats
    
    def clear_collection(self, delete_collection: bool = False) -> Dict[str, Any]:
        """
        Clear all points from the collection or delete the collection entirely.
        
        Args:
            delete_collection: If True, delete the entire collection; if False, just clear points
            
        Returns:
            Dictionary with operation result
        """
        if delete_collection:
            result = self._make_request("DELETE", f"/collections/{self.collection_name}")
            print(f"Collection '{self.collection_name}' deleted")
            return {"operation": "delete_collection", "result": result}
        else:
            # Delete all points by using a filter that matches everything
            delete_data = {
                "filter": {
                    "must": []
                }
            }
            result = self._make_request("POST", f"/collections/{self.collection_name}/points/delete", delete_data)
            print(f"All points cleared from collection '{self.collection_name}'")
            return {"operation": "clear_points", "result": result}
    
    def search_points(
        self,
        vector: List[float],
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict] = None,
        with_payload: bool = True,
        with_vector: bool = False
    ) -> Dict[str, Any]:
        """
        Search for similar vectors in the collection.
        
        Args:
            vector: Query vector
            limit: Number of results to return
            score_threshold: Minimum similarity score
            filter_conditions: Qdrant filter conditions
            with_payload: Include payload in results
            with_vector: Include vectors in results
        """
        search_data = {
            "vector": vector,
            "limit": limit,
            "with_payload": with_payload,
            "with_vector": with_vector
        }
        
        if score_threshold is not None:
            search_data["score_threshold"] = score_threshold
        
        if filter_conditions:
            search_data["filter"] = filter_conditions
        
        return self._make_request("POST", f"/collections/{self.collection_name}/points/search", search_data)
    
    def upsert_points(self, points: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Insert or update points in the collection.
        
        Args:
            points: List of points, each with 'id', 'vector', and 'payload'
        """
        upsert_data = {"points": points}
        return self._make_request("PUT", f"/collections/{self.collection_name}/points", upsert_data)
    
    def delete_points(self, point_ids: List[str]) -> Dict[str, Any]:
        """
        Delete specific points by their IDs.
        
        Args:
            point_ids: List of point IDs to delete
        """
        delete_data = {
            "points": point_ids
        }
        return self._make_request("POST", f"/collections/{self.collection_name}/points/delete", delete_data)
    
    def create_collection(
        self,
        collection_name: Optional[str] = None,
        vector_size: int = None,
        distance: str = "Cosine",
        shard_number: int = 1,
        on_disk_payload: bool = True
    ) -> Dict[str, Any]:
        """
        Create a new collection.
        
        Args:
            collection_name: Name of collection (defaults to instance collection_name)
            vector_size: Dimension of vectors (defaults to config)
            distance: Distance metric (Cosine, Euclid, Dot)
            shard_number: Number of shards
            on_disk_payload: Store payload on disk
        """
        coll_name = collection_name or self.collection_name
        if vector_size is None:
            vector_size = self.vector_size
        
        # Use Qdrant client if available
        if self.client:
            try:
                self.client.create_collection(
                    collection_name=coll_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE if distance == "Cosine" else Distance.DOT
                    ),
                )
                logger.info(f"Created collection '{coll_name}' via Qdrant client")
                return {"status": "success", "collection_name": coll_name}
            except Exception as e:
                logger.error(f"Failed to create collection via client: {e}")
        
        # Fallback to REST API
        create_data = {
            "vectors": {
                "size": vector_size,
                "distance": distance
            },
            "shard_number": shard_number,
            "on_disk_payload": on_disk_payload
        }
        return self._make_request("PUT", f"/collections/{coll_name}", create_data)
    
    def get_collection_stats(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive statistics about the collection."""
        coll_name = collection_name or self.collection_name
        info = self.inspect_collection(coll_name)
        
        if not info:
            return {
                "collection_name": coll_name,
                "status": "not_found",
                "points_count": 0
            }
        
        result = info.get("result", {})
        
        stats = {
            "collection_name": coll_name,
            "status": result.get("status"),
            "points_count": result.get("points_count", 0),
            "indexed_vectors_count": result.get("indexed_vectors_count", 0),
            "segments_count": result.get("segments_count", 0),
            "vector_size": result.get("config", {}).get("params", {}).get("vectors", {}).get("size"),
            "distance_metric": result.get("config", {}).get("params", {}).get("vectors", {}).get("distance"),
            "payload_schema": result.get("payload_schema", {})
        }
        
        return stats
    
    def print_stats(self, collection_name: Optional[str] = None):
        """Print collection statistics in a readable format."""
        coll_name = collection_name or self.collection_name
        try:
            stats = self.get_collection_stats(coll_name)
            print(f"\n{'='*60}")
            print(f"Collection: {stats['collection_name']}")
            print(f"{'='*60}")
            print(f"Status: {stats.get('status', 'unknown')}")
            print(f"Points Count: {stats.get('points_count', 0)}")
            print(f"Indexed Vectors: {stats.get('indexed_vectors_count', 0)}")
            print(f"Segments: {stats.get('segments_count', 0)}")
            print(f"Vector Size: {stats.get('vector_size', 'unknown')}")
            print(f"Distance Metric: {stats.get('distance_metric', 'unknown')}")
            print(f"{'='*60}\n")
        except Exception as e:
            print(f"Error getting stats for collection '{coll_name}': {e}")
    
    def print_search_results(self, results: Dict[str, Any], max_results: int = 5):
        """
        Pretty print search results with chunk information.
        
        Args:
            results: Search results from search_points
            max_results: Maximum number of results to display
        """
        search_results = results.get('result', [])
        
        if not search_results:
            print("No results found.")
            return
        
        print(f"\n{'='*60}")
        print(f"Search Results ({len(search_results)} found)")
        print(f"{'='*60}\n")
        
        for i, result in enumerate(search_results[:max_results], 1):
            score = result.get('score', 0)
            payload = result.get('payload', {})
            
            # Try new schema first, fallback to legacy
            source_type = payload.get('source_type', 'unknown')
            book_title = payload.get('book_title', payload.get('source', 'Unknown'))
            text_content = payload.get('text_content', '')
            
            # Build result header
            header = f"{i}. {book_title} ({source_type})"
            header += f" - Score: {score:.4f}"
            
            print(header)
            
            # Show metadata based on source type
            if source_type == 'quran':
                metadata = payload.get('source_metadata', {})
                verse_key = metadata.get('verse_key') or payload.get('verse_key', 'N/A')
                print(f"   Verse: {verse_key}")
            elif source_type == 'hadith':
                metadata = payload.get('source_metadata', {})
                hadith_num = metadata.get('hadith_number', 'N/A')
                authenticity = metadata.get('authenticity_grade', 'N/A')
                print(f"   Hadith #{hadith_num} - Grade: {authenticity}")
            
            # Show the matched content
            preview_len = 250
            if len(text_content) > preview_len:
                print(f"   Content: {text_content[:preview_len]}...")
            else:
                print(f"   Content: {text_content}")
            print()
        
        if len(search_results) > max_results:
            print(f"... and {len(search_results) - max_results} more results\n")


if __name__ == "__main__":
    # Example usage (for testing)
    qdrant_manager = QdrantManager()
    qdrant_manager.print_stats()
