"""
Code RAG System - 主流水线
整合所有组件，提供完整的代码RAG功能
"""

import os
import yaml
from pathlib import Path
from typing import List, Dict, Optional, Generator, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import json

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax

from chunker import CodeChunker, CodeChunk
from embedding import CodeEmbeddingManager, EmbeddingConfig
from vector_store import CodeVectorStore, VectorStoreConfig
from retriever import CodeRetriever, RetrieverConfig
from llm import LLMManager, LLMConfig, CodeRAGPrompts


console = Console()


@dataclass
class RAGConfig:
    """RAG系统总配置"""
    # 项目配置
    project_name: str = "default"                   # unset
    project_root: str = "."
    file_extensions: List[str] = None
    exclude_patterns: List[str] = None
    
    # 分块配置
    chunk_max_tokens: int = 1024
    chunk_overlap_tokens: int = 128
    
    # Embedding配置
    embedding_model: str = "BAAI/bge-large-zh-v1.5"
    embedding_device: str = "cuda"
    embedding_batch_size: int = 32                  # unset
    
    # 向量库配置
    vector_db_path: str = "./data/chroma_db"
    collection_name: str = "code_chunks"
    
    # 检索配置
    retrieval_top_k: int = 10
    use_rerank: bool = True
    rerank_model: str = "BAAI/bge-reranker-base"
    rerank_top_k: int = 5
    use_hybrid_search: bool = True
    
    # LLM配置
    llm_api_base: str = "http://localhost:8000/v1"
    llm_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    llm_max_tokens: int = 2048
    llm_temperature: float = 0.1
    
    def __post_init__(self):
        if self.file_extensions is None:
            self.file_extensions = [".cpp", ".h", ".hpp", ".c", ".cc"]
        if self.exclude_patterns is None:
            self.exclude_patterns = ["**/build/**", "**/third_party/**", "**/.git/**"]
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "RAGConfig":
        """从YAML文件加载配置"""
        with open(config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # 展平嵌套配置
        flat_config = {}
        
        if "project" in data:
            flat_config["project_name"] = data["project"].get("name", "default")
            flat_config["project_root"] = data["project"].get("root_path", ".")
            flat_config["file_extensions"] = data["project"].get("file_extensions")
            flat_config["exclude_patterns"] = data["project"].get("exclude_patterns")
        
        if "chunking" in data:
            flat_config["chunk_max_tokens"] = data["chunking"].get("max_tokens", 1024)
            flat_config["chunk_overlap_tokens"] = data["chunking"].get("overlap_tokens", 128)
        
        if "embedding" in data:
            flat_config["embedding_model"] = data["embedding"].get("model_name")
            flat_config["embedding_device"] = data["embedding"].get("device", "cuda")
            flat_config["embedding_batch_size"] = data["embedding"].get("batch_size", 32)
        
        if "vector_db" in data:
            flat_config["vector_db_path"] = data["vector_db"].get("persist_directory")
            flat_config["collection_name"] = data["vector_db"].get("collection_name")
        
        if "retrieval" in data:
            flat_config["retrieval_top_k"] = data["retrieval"].get("top_k", 10)
            flat_config["use_rerank"] = data["retrieval"].get("rerank", True)
            flat_config["rerank_model"] = data["retrieval"].get("rerank_model")
            flat_config["rerank_top_k"] = data["retrieval"].get("rerank_top_k", 5)
            if "hybrid_search" in data["retrieval"]:
                flat_config["use_hybrid_search"] = data["retrieval"]["hybrid_search"].get("enabled", True)
        
        if "llm" in data:
            flat_config["llm_api_base"] = data["llm"].get("api_base")
            flat_config["llm_model"] = data["llm"].get("model_name")
            flat_config["llm_max_tokens"] = data["llm"].get("max_tokens", 2048)
            flat_config["llm_temperature"] = data["llm"].get("temperature", 0.1)
        
        # 过滤None值
        flat_config = {k: v for k, v in flat_config.items() if v is not None}
        
        return cls(**flat_config)


class CodeRAGPipeline:
    """
    代码RAG主流水线
    提供索引构建、代码检索、问答等功能
    """
    
    def __init__(self, config: RAGConfig = None, config_path: str = None):
        if config_path:
            self.config = RAGConfig.from_yaml(config_path)
        else:
            self.config = config or RAGConfig()
        
        self._init_components()
        console.print(f"[green]✓[/green] Code RAG Pipeline initialized for: {self.config.project_name}")
    
    def _init_components(self):
        """初始化所有组件"""
        # 分块器
        self.chunker = CodeChunker(
            max_tokens=self.config.chunk_max_tokens,
            overlap_tokens=self.config.chunk_overlap_tokens
        )
        
        # Embedding管理器
        embedding_config = EmbeddingConfig(
            model_name=self.config.embedding_model,
            device=self.config.embedding_device,
            batch_size=self.config.embedding_batch_size
        )
        self.embedding_manager = CodeEmbeddingManager(embedding_config)
        
        # 向量存储
        store_config = VectorStoreConfig(
            persist_directory=self.config.vector_db_path,
            collection_name=self.config.collection_name
        )
        self.vector_store = CodeVectorStore(store_config)
        
        # 检索器
        retriever_config = RetrieverConfig(
            top_k=self.config.retrieval_top_k,
            rerank=self.config.use_rerank,
            rerank_model=self.config.rerank_model,
            rerank_top_k=self.config.rerank_top_k,
            hybrid_search=self.config.use_hybrid_search,
            device=self.config.embedding_device
        )
        self.retriever = CodeRetriever(
            self.embedding_manager,
            self.vector_store,
            retriever_config
        )
        
        # LLM管理器
        llm_config = LLMConfig(
            api_base=self.config.llm_api_base,
            model_name=self.config.llm_model,
            max_tokens=self.config.llm_max_tokens,
            temperature=self.config.llm_temperature
        )
        
        # 根据API基础URL自动检测后端类型
        if self.config.llm_api_base and "11434" in self.config.llm_api_base:
            backend = "ollama"
        else:
            backend = "openai"  # 默认为openai兼容接口
        
        self.llm_manager = LLMManager(llm_config, backend=backend)
    
    def build_index(
        self,
        root_path: str = None,
        incremental: bool = True,
        show_progress: bool = True
    ) -> Dict:
        """
        构建代码索引
        
        Args:
            root_path: 代码根目录
            incremental: 是否增量更新
            show_progress: 是否显示进度
        
        Returns:
            索引统计信息
        """
        root_path = root_path or self.config.project_root
        
        console.print(f"\n[bold]Building index for:[/bold] {root_path}")
        
        if not incremental:
            console.print("[yellow]Clearing existing index...[/yellow]")
            self.vector_store.clear()
        
        # 收集所有代码块
        all_chunks = []
        file_count = 0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            # 第一步：分块
            task1 = progress.add_task("[cyan]Chunking files...", total=None)
            
            for chunk in self.chunker.chunk_directory(
                root_path,
                extensions=self.config.file_extensions,
                exclude_patterns=self.config.exclude_patterns
            ):
                chunk_dict = {
                    "id": chunk.id,
                    "content": chunk.full_content if True else chunk.content,
                    "file_path": chunk.file_path,
                    "language": chunk.language,
                    "chunk_type": chunk.chunk_type,
                    "symbol_name": chunk.symbol_name,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                }
                all_chunks.append(chunk_dict)
                
                # 更新文件计数
                if not any(c["file_path"] == chunk.file_path for c in all_chunks[:-1]):
                    file_count += 1
                
                progress.update(task1, description=f"[cyan]Chunking... {file_count} files, {len(all_chunks)} chunks")
            
            progress.update(task1, completed=100)
            
            if not all_chunks:
                console.print("[yellow]No code chunks found.[/yellow]")
                return {"files": 0, "chunks": 0}
            
            # 第二步：生成Embedding
            task2 = progress.add_task("[cyan]Generating embeddings...", total=len(all_chunks))
            
            # 分批处理
            batch_size = self.config.embedding_batch_size
            all_embeddings = []
            
            for i in range(0, len(all_chunks), batch_size):
                batch = all_chunks[i:i + batch_size]
                embeddings = self.embedding_manager.encode_code(batch, show_progress=False)
                all_embeddings.append(embeddings)
                progress.update(task2, advance=len(batch))
            
            import numpy as np
            embeddings = np.vstack(all_embeddings)
            
            # 第三步：存储到向量库
            task3 = progress.add_task("[cyan]Storing to vector DB...", total=100)
            self.vector_store.add_chunks(all_chunks, embeddings)
            progress.update(task3, completed=100)
        
        stats = {
            "files": file_count,
            "chunks": len(all_chunks),
            "timestamp": datetime.now().isoformat()
        }
        
        # 保存索引元数据
        meta_path = Path(self.config.vector_db_path) / "index_meta.json"
        with open(meta_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        console.print(f"\n[green]✓[/green] Index built: {stats['files']} files, {stats['chunks']} chunks")
        
        return stats
    
    def search(
        self,
        query: str,
        top_k: int = None,
        file_filter: str = None,
        language_filter: str = None
    ) -> List[Dict]:
        """
        搜索代码
        
        Args:
            query: 搜索查询
            top_k: 返回数量
            file_filter: 文件路径过滤
            language_filter: 语言过滤
        
        Returns:
            搜索结果列表
        """
        results = self.retriever.retrieve(
            query,
            top_k=top_k or self.config.retrieval_top_k,
            file_filter=file_filter,
            language_filter=language_filter
        )
        
        return results
    
    def ask(
        self,
        question: str,
        top_k: int = None,
        stream: bool = False,
        show_context: bool = False
    ) -> Union[str, Generator[str, None, None]]:
        """
        询问代码相关问题
        
        Args:
            question: 问题
            top_k: 检索上下文数量
            stream: 是否流式输出
            show_context: 是否显示检索到的上下文
        
        Returns:
            回答或生成器
        """
        # 检索相关代码
        context = self.search(question, top_k=top_k)
        
        if show_context:
            self._display_context(context)
        
        if not context:
            return "未找到相关代码上下文，无法回答问题。"
        
        # 生成回答
        return self.llm_manager.answer_code_question(question, context, stream=stream)
    
    def _display_context(self, context: List[Dict]):
        """显示检索到的上下文"""
        console.print("\n[bold]Retrieved Context:[/bold]")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Score", style="cyan", width=8)
        table.add_column("File", style="green")
        table.add_column("Type", style="yellow")
        table.add_column("Symbol", style="blue")
        table.add_column("Lines", style="dim")
        
        for ctx in context:
            meta = ctx.get("metadata", {})
            score = ctx.get("rerank_score", ctx.get("score", 0))
            table.add_row(
                f"{score:.3f}",
                meta.get("file_path", "?")[-40:],
                meta.get("chunk_type", "?"),
                meta.get("symbol_name", "-"),
                f"{meta.get('start_line', '?')}-{meta.get('end_line', '?')}"
            )
        
        console.print(table)
    
    def explain_code(
        self,
        file_path: str,
        start_line: int = None,
        end_line: int = None,
        stream: bool = False
    ) -> Union[str, Generator[str, None, None]]:
        """
        解释指定代码
        
        Args:
            file_path: 文件路径
            start_line: 起始行
            end_line: 结束行
            stream: 是否流式输出
        
        Returns:
            解释文本
        """
        # 读取代码
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
        except Exception as e:
            return f"无法读取文件: {e}"
        
        # 提取指定行
        if start_line and end_line:
            code_lines = lines[start_line - 1:end_line]
        else:
            code_lines = lines
            start_line = 1
            end_line = len(lines)
        
        code = ''.join(code_lines)
        
        # 检测语言
        ext = Path(file_path).suffix.lower()
        lang_map = {'.cpp': 'cpp', '.h': 'cpp', '.c': 'c', '.py': 'python', '.js': 'javascript'}
        language = lang_map.get(ext, 'text')
        
        # 生成提示
        prompt = CodeRAGPrompts.format_code_explain(
            code=code,
            file_path=file_path,
            language=language,
            start_line=start_line,
            end_line=end_line
        )
        
        if stream:
            return self.llm_manager.generate_stream(prompt)
        else:
            return self.llm_manager.generate(prompt)
    
    def get_stats(self) -> Dict:
        """获取系统统计信息"""
        db_stats = self.vector_store.get_stats()
        
        # 尝试读取索引元数据
        meta_path = Path(self.config.vector_db_path) / "index_meta.json"
        index_meta = {}
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                index_meta = json.load(f)
        
        return {
            "project": self.config.project_name,
            "vector_db": db_stats,
            "index": index_meta,
            "config": {
                "embedding_model": self.config.embedding_model,
                "llm_model": self.config.llm_model,
                "retrieval_top_k": self.config.retrieval_top_k,
            }
        }
    
    def interactive_session(self):
        """启动交互式会话"""
        console.print(Panel.fit(
            "[bold green]Code RAG Interactive Session[/bold green]\n"
            "Commands: /search <query> | /explain <file> | /stats | /quit",
            border_style="green"
        ))
        
        while True:
            try:
                user_input = console.input("\n[bold blue]You:[/bold blue] ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['/quit', '/exit', '/q']:
                    console.print("[yellow]Goodbye![/yellow]")
                    break
                
                if user_input.startswith('/search '):
                    query = user_input[8:]
                    results = self.search(query)
                    self._display_context(results)
                    continue
                
                if user_input.startswith('/explain '):
                    parts = user_input[9:].split()
                    file_path = parts[0]
                    console.print(f"\n[bold green]Explanation:[/bold green]")
                    for chunk in self.explain_code(file_path, stream=True):
                        console.print(chunk, end="")
                    console.print()
                    continue
                
                if user_input == '/stats':
                    stats = self.get_stats()
                    console.print_json(data=stats)
                    continue
                
                # 默认：问答模式
                console.print(f"\n[bold green]Assistant:[/bold green]")
                for chunk in self.ask(user_input, stream=True, show_context=True):
                    console.print(chunk, end="")
                console.print()
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Use /quit to exit[/yellow]")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")


# 快捷函数
def create_pipeline(config_path: str = None, **kwargs) -> CodeRAGPipeline:
    """创建RAG流水线的快捷函数"""
    if config_path:
        return CodeRAGPipeline(config_path=config_path)
    else:
        config = RAGConfig(**kwargs)
        return CodeRAGPipeline(config=config)


# 使用示例
if __name__ == "__main__":
    # 从配置文件创建
    # pipeline = create_pipeline(config_path="config.yaml")
    
    # 或直接指定参数
    pipeline = create_pipeline(
        project_name="my-project",
        project_root="/path/to/code",
        embedding_model="BAAI/bge-large-zh-v1.5",
        llm_api_base="http://localhost:8000/v1",
        llm_model="Qwen/Qwen2.5-Coder-7B-Instruct"
    )
    
    # 构建索引
    # pipeline.build_index()
    
    # 搜索代码
    # results = pipeline.search("排序算法")
    
    # 问答
    # answer = pipeline.ask("这个项目中有哪些排序算法的实现？")
    
    # 交互式会话
    # pipeline.interactive_session()
