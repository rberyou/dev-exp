"""
Code RAG System - 向量数据库管理
基于 ChromaDB 实现，支持持久化和混合检索
"""

import os
import json
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import numpy as np


@dataclass
class VectorStoreConfig:
    """向量数据库配置"""
    persist_directory: str = "./data/chroma_db"
    collection_name: str = "code_chunks"
    distance_metric: str = "cosine"  # cosine, l2, ip


class VectorStore:
    """
    向量数据库管理器
    封装 ChromaDB 操作
    """
    
    def __init__(self, config: VectorStoreConfig = None):
        self.config = config or VectorStoreConfig()
        self.client = None
        self.collection = None
        self._init_db()
    
    def _init_db(self):
        """初始化数据库"""
        # 确保目录存在
        Path(self.config.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # 创建持久化客户端
        self.client = chromadb.PersistentClient(
            path=self.config.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # 获取或创建集合
        self.collection = self.client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={"hnsw:space": self.config.distance_metric}
        )
        
        print(f"Vector store initialized. Collection: {self.config.collection_name}")
        print(f"Existing documents: {self.collection.count()}")
    
    def _generate_id(self, content: str, file_path: str) -> str:
        """生成唯一ID"""
        unique_str = f"{file_path}:{content[:100]}"
        return hashlib.md5(unique_str.encode()).hexdigest()
    
    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict] = None
    ):
        """
        添加向量到数据库
        
        Args:
            ids: 唯一ID列表
            embeddings: 嵌入向量列表
            documents: 原始文档列表
            metadatas: 元数据列表
        """
        if metadatas is None:
            metadatas = [{}] * len(ids)
        
        # ChromaDB metadata只支持基本类型，需要转换
        clean_metadatas = []
        for meta in metadatas:
            clean_meta = {}
            for k, v in meta.items():
                if isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                elif isinstance(v, list):
                    clean_meta[k] = json.dumps(v)
                else:
                    clean_meta[k] = str(v)
            clean_metadatas.append(clean_meta)
        
        # 分批添加（ChromaDB单次限制）
        batch_size = 5000
        for i in range(0, len(ids), batch_size):
            end_idx = min(i + batch_size, len(ids))
            self.collection.add(
                ids=ids[i:end_idx],
                embeddings=embeddings[i:end_idx],
                documents=documents[i:end_idx],
                metadatas=clean_metadatas[i:end_idx]
            )
        
        print(f"Added {len(ids)} documents. Total: {self.collection.count()}")
    
    def query(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        where: Dict = None,
        where_document: Dict = None,
        include: List[str] = None
    ) -> Dict:
        """
        查询相似向量
        
        Args:
            query_embedding: 查询向量
            top_k: 返回数量
            where: 元数据过滤条件
            where_document: 文档内容过滤条件
            include: 返回字段 ["documents", "metadatas", "distances", "embeddings"]
        
        Returns:
            查询结果字典
        """
        if include is None:
            include = ["documents", "metadatas", "distances"]
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            where_document=where_document,
            include=include
        )
        
        # 展平结果
        return {
            "ids": results["ids"][0] if results["ids"] else [],
            "documents": results["documents"][0] if results.get("documents") else [],
            "metadatas": results["metadatas"][0] if results.get("metadatas") else [],
            "distances": results["distances"][0] if results.get("distances") else [],
        }
    
    def search_by_metadata(
        self,
        where: Dict,
        limit: int = 100
    ) -> Dict:
        """按元数据搜索"""
        results = self.collection.get(
            where=where,
            limit=limit,
            include=["documents", "metadatas"]
        )
        return results
    
    def get_by_ids(self, ids: List[str]) -> Dict:
        """按ID获取"""
        return self.collection.get(
            ids=ids,
            include=["documents", "metadatas", "embeddings"]
        )
    
    def update(
        self,
        ids: List[str],
        embeddings: List[List[float]] = None,
        documents: List[str] = None,
        metadatas: List[Dict] = None
    ):
        """更新文档"""
        self.collection.update(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
    
    def delete(self, ids: List[str] = None, where: Dict = None):
        """删除文档"""
        if ids:
            self.collection.delete(ids=ids)
        elif where:
            self.collection.delete(where=where)
    
    def clear(self):
        """清空集合"""
        self.client.delete_collection(self.config.collection_name)
        self.collection = self.client.create_collection(
            name=self.config.collection_name,
            metadata={"hnsw:space": self.config.distance_metric}
        )
        print("Collection cleared.")
    
    def count(self) -> int:
        """获取文档数量"""
        return self.collection.count()
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            "collection_name": self.config.collection_name,
            "document_count": self.collection.count(),
            "persist_directory": self.config.persist_directory,
        }


class CodeVectorStore(VectorStore):
    """
    代码专用向量数据库
    增加代码相关的便捷方法
    """
    
    def add_chunks(
        self,
        chunks: List[Dict],
        embeddings: np.ndarray
    ):
        """
        添加代码块
        
        Args:
            chunks: 代码块列表（包含 id, content, file_path, chunk_type 等）
            embeddings: 对应的嵌入向量
        """
        ids = []
        documents = []
        metadatas = []
        
        for chunk in chunks:
            chunk_id = chunk.get("id") or self._generate_id(
                chunk.get("content", ""),
                chunk.get("file_path", "")
            )
            ids.append(chunk_id)
            documents.append(chunk.get("content", ""))
            
            # 构建元数据
            meta = {
                "file_path": chunk.get("file_path", ""),
                "language": chunk.get("language", ""),
                "chunk_type": chunk.get("chunk_type", ""),
                "symbol_name": chunk.get("symbol_name", ""),
                "start_line": chunk.get("start_line", 0),
                "end_line": chunk.get("end_line", 0),
            }
            metadatas.append(meta)
        
        self.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas
        )
    
    def search_code(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        file_filter: str = None,
        language_filter: str = None,
        chunk_type_filter: str = None
    ) -> List[Dict]:
        """
        搜索代码
        
        Args:
            query_embedding: 查询向量
            top_k: 返回数量
            file_filter: 文件路径过滤（支持前缀匹配）
            language_filter: 语言过滤
            chunk_type_filter: 代码块类型过滤
        
        Returns:
            搜索结果列表
        """
        # 构建过滤条件
        where = {}
        if language_filter:
            where["language"] = language_filter
        if chunk_type_filter:
            where["chunk_type"] = chunk_type_filter
        
        # ChromaDB 暂不支持前缀匹配，先获取更多结果再过滤
        results = self.query(
            query_embedding=query_embedding.flatten().tolist(),
            top_k=top_k * 3 if file_filter else top_k,
            where=where if where else None
        )
        
        # 组装结果
        search_results = []
        for i, doc_id in enumerate(results["ids"]):
            result = {
                "id": doc_id,
                "content": results["documents"][i] if results["documents"] else "",
                "metadata": results["metadatas"][i] if results["metadatas"] else {},
                "distance": results["distances"][i] if results["distances"] else 0,
                "score": 1 - results["distances"][i] if results["distances"] else 1,  # 转为相似度
            }
            
            # 文件路径过滤
            if file_filter:
                file_path = result["metadata"].get("file_path", "")
                if not file_path.startswith(file_filter):
                    continue
            
            search_results.append(result)
            
            if len(search_results) >= top_k:
                break
        
        return search_results
    
    def get_file_chunks(self, file_path: str) -> List[Dict]:
        """获取指定文件的所有代码块"""
        results = self.search_by_metadata(
            where={"file_path": file_path},
            limit=1000
        )
        
        chunks = []
        for i, doc_id in enumerate(results["ids"]):
            chunks.append({
                "id": doc_id,
                "content": results["documents"][i] if results.get("documents") else "",
                "metadata": results["metadatas"][i] if results.get("metadatas") else {},
            })
        
        # 按行号排序
        chunks.sort(key=lambda x: x["metadata"].get("start_line", 0))
        return chunks
    
    def delete_file(self, file_path: str):
        """删除指定文件的所有代码块"""
        self.delete(where={"file_path": file_path})
        print(f"Deleted chunks for: {file_path}")
    
    def get_languages(self) -> List[str]:
        """获取所有语言类型"""
        # ChromaDB 暂不支持 distinct，获取一批数据统计
        results = self.collection.get(limit=10000, include=["metadatas"])
        languages = set()
        for meta in results.get("metadatas", []):
            if meta and "language" in meta:
                languages.add(meta["language"])
        return list(languages)
    
    def get_files(self) -> List[str]:
        """获取所有文件路径"""
        results = self.collection.get(limit=10000, include=["metadatas"])
        files = set()
        for meta in results.get("metadatas", []):
            if meta and "file_path" in meta:
                files.add(meta["file_path"])
        return sorted(list(files))


# 使用示例
if __name__ == "__main__":
    # 创建向量存储
    config = VectorStoreConfig(
        persist_directory="./data/chroma_db",
        collection_name="code_chunks"
    )
    
    store = CodeVectorStore(config)
    
    # 打印统计信息
    print(store.get_stats())
