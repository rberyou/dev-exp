"""
Code RAG System - Embedding 管理器
支持多种嵌入模型，针对代码优化
"""

import os
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
import numpy as np
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer


@dataclass
class EmbeddingConfig:
    """Embedding配置"""
    model_name: str = "BAAI/bge-large-zh-v1.5"
    device: str = "cuda"
    batch_size: int = 32
    normalize: bool = True
    max_length: int = 512
    # cache_dir: str = "./models"


class EmbeddingManager:
    """
    Embedding管理器
    支持多种模型：BGE、CodeBERT、E5等
    """
    
    # 支持的模型类型
    SUPPORTED_MODELS = {
        # BGE系列 - 中文优化
        "BAAI/bge-large-zh-v1.5": {"type": "sentence_transformer", "dim": 1024},
        "BAAI/bge-base-zh-v1.5": {"type": "sentence_transformer", "dim": 768},
        "BAAI/bge-small-zh-v1.5": {"type": "sentence_transformer", "dim": 512},
        
        # BGE系列 - 英文/通用
        "BAAI/bge-large-en-v1.5": {"type": "sentence_transformer", "dim": 1024},
        "BAAI/bge-base-en-v1.5": {"type": "sentence_transformer", "dim": 768},
        
        # 代码专用模型
        "microsoft/codebert-base": {"type": "huggingface", "dim": 768},
        "microsoft/graphcodebert-base": {"type": "huggingface", "dim": 768},
        "microsoft/unixcoder-base": {"type": "huggingface", "dim": 768},
        
        # E5系列
        "intfloat/e5-large-v2": {"type": "sentence_transformer", "dim": 1024},
        "intfloat/multilingual-e5-large": {"type": "sentence_transformer", "dim": 1024},
    }
    
    def __init__(self, config: EmbeddingConfig = None):
        self.config = config or EmbeddingConfig()
        self.model = None
        self.tokenizer = None
        self.model_info = None
        self._load_model()
    
    def _load_model(self):
        """加载模型"""
        model_name = self.config.model_name
        
        # 确定模型类型
        if model_name in self.SUPPORTED_MODELS:
            self.model_info = self.SUPPORTED_MODELS[model_name]
        else:
            # 默认使用sentence_transformer
            self.model_info = {"type": "sentence_transformer", "dim": 768}
        
        print(f"Loading embedding model: {model_name}")
        
        if self.model_info["type"] == "sentence_transformer":
            self.model = SentenceTransformer(
                model_name,
                device=self.config.device
            )
        else:
            # HuggingFace模型
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=self.config.cache_dir
            )
            self.model = AutoModel.from_pretrained(
                model_name,
                cache_dir=self.config.cache_dir
            ).to(self.config.device)
            self.model.eval()
        
        print(f"Model loaded. Embedding dimension: {self.model_info['dim']}")
    
    @property
    def dimension(self) -> int:
        """获取embedding维度"""
        return self.model_info["dim"]
    
    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling for HuggingFace models"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
    
    def _encode_huggingface(self, texts: List[str]) -> np.ndarray:
        """使用HuggingFace模型编码"""
        all_embeddings = []
        
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt"
            ).to(self.config.device)
            
            with torch.no_grad():
                outputs = self.model(**encoded)
                embeddings = self._mean_pooling(outputs, encoded["attention_mask"])
                
                if self.config.normalize:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def encode(
        self,
        texts: Union[str, List[str]],
        show_progress: bool = True,
        prefix: str = ""
    ) -> np.ndarray:
        """
        编码文本为向量
        
        Args:
            texts: 单个文本或文本列表
            show_progress: 是否显示进度条
            prefix: 文本前缀（某些模型需要，如E5）
        
        Returns:
            np.ndarray: 嵌入向量，shape为 (n, dim)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # 添加前缀（如果需要）
        if prefix:
            texts = [prefix + t for t in texts]
        
        if self.model_info["type"] == "sentence_transformer":
            embeddings = self.model.encode(
                texts,
                batch_size=self.config.batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=self.config.normalize
            )
        else:
            embeddings = self._encode_huggingface(texts)
        
        return np.array(embeddings)
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        编码查询文本
        某些模型对query和document使用不同的前缀
        """
        # BGE模型的query前缀
        if "bge" in self.config.model_name.lower():
            prefix = "为这个句子生成表示以用于检索相关文章：" if "zh" in self.config.model_name else "Represent this sentence for searching relevant passages: "
        # E5模型的query前缀
        elif "e5" in self.config.model_name.lower():
            prefix = "query: "
        else:
            prefix = ""
        
        return self.encode(query, show_progress=False, prefix=prefix)
    
    def encode_documents(self, documents: List[str], show_progress: bool = True) -> np.ndarray:
        """
        编码文档文本
        """
        # E5模型的document前缀
        if "e5" in self.config.model_name.lower():
            prefix = "passage: "
        else:
            prefix = ""
        
        return self.encode(documents, show_progress=show_progress, prefix=prefix)
    
    def similarity(self, query_embedding: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
        """计算相似度分数"""
        # 余弦相似度（如果已归一化，点积即为余弦相似度）
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        scores = np.dot(doc_embeddings, query_embedding.T).flatten()
        return scores


class CodeEmbeddingManager(EmbeddingManager):
    """
    代码专用Embedding管理器
    增强代码相关的处理
    """
    
    def __init__(self, config: EmbeddingConfig = None):
        # 默认使用代码优化模型
        if config is None:
            config = EmbeddingConfig(model_name="BAAI/bge-large-zh-v1.5")
        super().__init__(config)
    
    def encode_code(
        self,
        code_chunks: List[Dict],
        include_context: bool = True,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        编码代码块
        
        Args:
            code_chunks: 代码块列表，每个包含 content, file_path, chunk_type 等
            include_context: 是否包含上下文信息
            show_progress: 是否显示进度
        
        Returns:
            代码块的嵌入向量
        """
        texts = []
        
        for chunk in code_chunks:
            if include_context:
                # 构建包含上下文的文本
                text = self._build_context_text(chunk)
            else:
                text = chunk.get("content", "")
            texts.append(text)
        
        return self.encode_documents(texts, show_progress=show_progress)
    
    def _build_context_text(self, chunk: Dict) -> str:
        """构建包含上下文的文本"""
        parts = []
        
        # 文件路径
        if "file_path" in chunk:
            parts.append(f"File: {chunk['file_path']}")
        
        # 代码类型
        if "chunk_type" in chunk:
            parts.append(f"Type: {chunk['chunk_type']}")
        
        # 符号名称
        if "symbol_name" in chunk and chunk["symbol_name"]:
            parts.append(f"Symbol: {chunk['symbol_name']}")
        
        # 代码内容
        parts.append(chunk.get("content", ""))
        
        return "\n".join(parts)
    
    def encode_code_query(self, query: str, query_type: str = "general") -> np.ndarray:
        """
        编码代码查询
        
        Args:
            query: 查询文本
            query_type: 查询类型 (general, function, class, bug, explain)
        """
        # 根据查询类型添加提示
        prompts = {
            "general": "",
            "function": "Find the function that: ",
            "class": "Find the class that: ",
            "bug": "Find code related to this bug: ",
            "explain": "Find code to explain: ",
            "similar": "Find similar code to: ",
        }
        
        prefix = prompts.get(query_type, "")
        enhanced_query = prefix + query
        
        return self.encode_query(enhanced_query)


# 使用示例
if __name__ == "__main__":
    # 创建embedding管理器
    config = EmbeddingConfig(
        model_name="BAAI/bge-large-zh-v1.5",
        device="cuda",
        batch_size=32
    )
    
    manager = CodeEmbeddingManager(config)
    
    # 测试编码
    test_code = [
        {
            "content": "void quickSort(int arr[], int low, int high) { ... }",
            "file_path": "sort.cpp",
            "chunk_type": "function_definition",
            "symbol_name": "quickSort"
        }
    ]
    
    embeddings = manager.encode_code(test_code)
    print(f"Embedding shape: {embeddings.shape}")
    
    # 测试查询
    query_embedding = manager.encode_code_query("排序算法实现", query_type="function")
    print(f"Query embedding shape: {query_embedding.shape}")
