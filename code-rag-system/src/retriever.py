"""
Code RAG System - 检索器
支持向量检索、关键词检索、重排序
"""

import re
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass
from collections import Counter
import numpy as np

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class RetrieverConfig:
    """检索器配置"""
    top_k: int = 10
    rerank: bool = True
    rerank_model: str = "BAAI/bge-reranker-base"
    rerank_top_k: int = 5
    hybrid_search: bool = True
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3
    device: str = "cuda"
    cache_dir: str = "./models"


class Reranker:
    """
    重排序器
    使用交叉编码器对检索结果进行精排
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: str = "cuda",
        cache_dir: str = "./models"
    ):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, cache_dir=cache_dir
        ).to(device)
        self.model.eval()
        print(f"Reranker loaded: {model_name}")
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = None
    ) -> List[Tuple[int, float]]:
        """
        对文档进行重排序
        
        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回top_k个结果
        
        Returns:
            [(原始索引, 分数), ...] 按分数降序排列
        """
        if not documents:
            return []
        
        # 构建输入对
        pairs = [[query, doc] for doc in documents]
        
        # 分批处理
        batch_size = 32
        all_scores = []
        
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = outputs.logits.squeeze(-1).cpu().numpy()
                if scores.ndim == 0:
                    scores = [float(scores)]
                all_scores.extend(scores.tolist() if hasattr(scores, 'tolist') else scores)
        
        # 排序
        indexed_scores = list(enumerate(all_scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        if top_k:
            indexed_scores = indexed_scores[:top_k]
        
        return indexed_scores


class KeywordSearcher:
    """
    关键词搜索器
    使用BM25风格的关键词匹配
    """
    
    def __init__(self):
        self.idf_cache = {}
    
    def _tokenize(self, text: str) -> List[str]:
        """简单分词"""
        # 保留代码相关的token
        text = text.lower()
        # 分割驼峰命名
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        # 分割下划线命名
        text = text.replace('_', ' ')
        # 提取单词
        tokens = re.findall(r'\b[a-z][a-z0-9]*\b', text)
        return tokens
    
    def _compute_idf(self, documents: List[str]) -> Dict[str, float]:
        """计算IDF"""
        doc_count = len(documents)
        term_doc_count = Counter()
        
        for doc in documents:
            terms = set(self._tokenize(doc))
            for term in terms:
                term_doc_count[term] += 1
        
        idf = {}
        for term, count in term_doc_count.items():
            idf[term] = np.log((doc_count + 1) / (count + 1)) + 1
        
        return idf
    
    def search(
        self,
        query: str,
        documents: List[str],
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        关键词搜索
        
        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回数量
        
        Returns:
            [(原始索引, 分数), ...] 按分数降序排列
        """
        if not documents:
            return []
        
        # 分词
        query_terms = self._tokenize(query)
        if not query_terms:
            return [(i, 0.0) for i in range(min(top_k, len(documents)))]
        
        # 计算IDF
        idf = self._compute_idf(documents)
        
        # 计算每个文档的分数
        scores = []
        for i, doc in enumerate(documents):
            doc_terms = self._tokenize(doc)
            doc_term_count = Counter(doc_terms)
            doc_len = len(doc_terms)
            
            score = 0.0
            for term in query_terms:
                if term in doc_term_count:
                    tf = doc_term_count[term] / (doc_len + 1)
                    score += tf * idf.get(term, 1.0)
            
            scores.append((i, score))
        
        # 排序
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class CodeRetriever:
    """
    代码检索器
    整合向量检索、关键词检索和重排序
    """
    
    def __init__(
        self,
        embedding_manager,
        vector_store,
        config: RetrieverConfig = None
    ):
        self.config = config or RetrieverConfig()
        self.embedding_manager = embedding_manager
        self.vector_store = vector_store
        
        # 初始化组件
        self.keyword_searcher = KeywordSearcher() if self.config.hybrid_search else None
        self.reranker = None
        if self.config.rerank:
            self.reranker = Reranker(
                model_name=self.config.rerank_model,
                device=self.config.device,
                cache_dir=self.config.cache_dir
            )
    
    def retrieve(
        self,
        query: str,
        top_k: int = None,
        file_filter: str = None,
        language_filter: str = None,
        chunk_type_filter: str = None,
        use_rerank: bool = None
    ) -> List[Dict]:
        """
        检索相关代码
        
        Args:
            query: 查询文本
            top_k: 返回数量
            file_filter: 文件路径过滤
            language_filter: 语言过滤
            chunk_type_filter: 代码块类型过滤
            use_rerank: 是否使用重排序
        
        Returns:
            检索结果列表
        """
        top_k = top_k or self.config.top_k
        use_rerank = use_rerank if use_rerank is not None else self.config.rerank
        
        # 获取更多候选用于重排序
        initial_k = top_k * 3 if use_rerank else top_k
        
        # 向量检索
        query_embedding = self.embedding_manager.encode_code_query(query)
        semantic_results = self.vector_store.search_code(
            query_embedding=query_embedding,
            top_k=initial_k,
            file_filter=file_filter,
            language_filter=language_filter,
            chunk_type_filter=chunk_type_filter
        )
        
        # 混合检索
        if self.config.hybrid_search and self.keyword_searcher:
            results = self._hybrid_search(query, semantic_results, initial_k)
        else:
            results = semantic_results
        
        # 重排序
        if use_rerank and self.reranker and results:
            results = self._rerank_results(query, results, top_k)
        else:
            results = results[:top_k]
        
        return results
    
    def _hybrid_search(
        self,
        query: str,
        semantic_results: List[Dict],
        top_k: int
    ) -> List[Dict]:
        """混合检索（语义+关键词）"""
        if not semantic_results:
            return []
        
        # 提取文档内容
        documents = [r["content"] for r in semantic_results]
        
        # 关键词搜索
        keyword_scores = self.keyword_searcher.search(query, documents, len(documents))
        keyword_score_map = {idx: score for idx, score in keyword_scores}
        
        # 归一化语义分数
        semantic_scores = [r["score"] for r in semantic_results]
        max_semantic = max(semantic_scores) if semantic_scores else 1
        min_semantic = min(semantic_scores) if semantic_scores else 0
        
        # 归一化关键词分数
        keyword_scores_list = [s for _, s in keyword_scores]
        max_keyword = max(keyword_scores_list) if keyword_scores_list else 1
        min_keyword = min(keyword_scores_list) if keyword_scores_list else 0
        
        # 计算混合分数
        for i, result in enumerate(semantic_results):
            # 归一化
            norm_semantic = (result["score"] - min_semantic) / (max_semantic - min_semantic + 1e-9)
            keyword_score = keyword_score_map.get(i, 0)
            norm_keyword = (keyword_score - min_keyword) / (max_keyword - min_keyword + 1e-9)
            
            # 加权混合
            result["hybrid_score"] = (
                self.config.semantic_weight * norm_semantic +
                self.config.keyword_weight * norm_keyword
            )
            result["semantic_score"] = result["score"]
            result["keyword_score"] = keyword_score
        
        # 按混合分数排序
        semantic_results.sort(key=lambda x: x.get("hybrid_score", 0), reverse=True)
        
        return semantic_results[:top_k]
    
    def _rerank_results(
        self,
        query: str,
        results: List[Dict],
        top_k: int
    ) -> List[Dict]:
        """重排序结果"""
        documents = [r["content"] for r in results]
        reranked = self.reranker.rerank(query, documents, top_k=self.config.rerank_top_k)
        
        # 重新排列结果
        reranked_results = []
        for original_idx, rerank_score in reranked:
            result = results[original_idx].copy()
            result["rerank_score"] = rerank_score
            result["original_rank"] = original_idx
            reranked_results.append(result)
        
        return reranked_results[:top_k]
    
    def retrieve_with_context(
        self,
        query: str,
        top_k: int = None,
        context_lines: int = 10,
        **kwargs
    ) -> List[Dict]:
        """
        检索并扩展上下文
        
        Args:
            query: 查询文本
            top_k: 返回数量
            context_lines: 上下文扩展行数
            **kwargs: 其他过滤参数
        
        Returns:
            带扩展上下文的检索结果
        """
        results = self.retrieve(query, top_k=top_k, **kwargs)
        
        # 扩展上下文（从同一文件获取相邻代码块）
        for result in results:
            file_path = result["metadata"].get("file_path")
            if not file_path:
                continue
            
            # 获取同文件的所有块
            file_chunks = self.vector_store.get_file_chunks(file_path)
            current_start = result["metadata"].get("start_line", 0)
            current_end = result["metadata"].get("end_line", 0)
            
            # 找到相邻的块
            context_before = []
            context_after = []
            
            for chunk in file_chunks:
                chunk_end = chunk["metadata"].get("end_line", 0)
                chunk_start = chunk["metadata"].get("start_line", 0)
                
                if chunk_end < current_start and current_start - chunk_end <= context_lines:
                    context_before.append(chunk["content"])
                elif chunk_start > current_end and chunk_start - current_end <= context_lines:
                    context_after.append(chunk["content"])
            
            result["context_before"] = "\n".join(context_before[-2:])  # 最多2个前置块
            result["context_after"] = "\n".join(context_after[:2])  # 最多2个后置块
        
        return results


class HyDERetriever(CodeRetriever):
    """
    HyDE (Hypothetical Document Embeddings) 检索器
    让LLM先生成假设性答案，再用其进行检索
    """
    
    def __init__(
        self,
        embedding_manager,
        vector_store,
        llm_client,
        config: RetrieverConfig = None
    ):
        super().__init__(embedding_manager, vector_store, config)
        self.llm_client = llm_client
    
    def _generate_hypothetical_document(self, query: str) -> str:
        """生成假设性文档"""
        prompt = f"""Based on the following question about code, write a hypothetical code snippet or documentation that would answer this question. Write only the code/doc, no explanations.

Question: {query}

Hypothetical code/documentation:"""
        
        response = self.llm_client.generate(prompt, max_tokens=500)
        return response
    
    def retrieve(
        self,
        query: str,
        use_hyde: bool = True,
        **kwargs
    ) -> List[Dict]:
        """
        使用HyDE进行检索
        
        Args:
            query: 查询文本
            use_hyde: 是否使用HyDE
            **kwargs: 其他参数
        
        Returns:
            检索结果
        """
        if use_hyde:
            # 生成假设性文档
            hypothetical_doc = self._generate_hypothetical_document(query)
            
            # 使用假设性文档的embedding进行检索
            hyde_embedding = self.embedding_manager.encode_documents([hypothetical_doc])
            
            # 同时使用原始查询embedding
            query_embedding = self.embedding_manager.encode_code_query(query)
            
            # 混合两种embedding
            combined_embedding = (hyde_embedding + query_embedding) / 2
            
            top_k = kwargs.get("top_k", self.config.top_k)
            results = self.vector_store.search_code(
                query_embedding=combined_embedding,
                top_k=top_k * 3 if self.config.rerank else top_k,
                **{k: v for k, v in kwargs.items() if k != "top_k"}
            )
            
            # 重排序
            if self.config.rerank and self.reranker:
                results = self._rerank_results(query, results, top_k)
            
            return results
        else:
            return super().retrieve(query, **kwargs)


# 使用示例
if __name__ == "__main__":
    from embedding import CodeEmbeddingManager, EmbeddingConfig
    from vector_store import CodeVectorStore, VectorStoreConfig
    
    # 初始化组件
    embedding_config = EmbeddingConfig(model_name="BAAI/bge-large-zh-v1.5")
    embedding_manager = CodeEmbeddingManager(embedding_config)
    
    store_config = VectorStoreConfig(persist_directory="./data/chroma_db")
    vector_store = CodeVectorStore(store_config)
    
    retriever_config = RetrieverConfig(
        top_k=10,
        rerank=True,
        hybrid_search=True
    )
    
    retriever = CodeRetriever(embedding_manager, vector_store, retriever_config)
    
    # 测试检索
    # results = retriever.retrieve("排序算法实现")
    # for r in results:
    #     print(f"[{r['score']:.3f}] {r['metadata'].get('file_path')} - {r['metadata'].get('symbol_name')}")
