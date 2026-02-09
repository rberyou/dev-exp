#!/usr/bin/env python3
"""
Code RAG System - 完整使用示例
演示如何为大型项目构建RAG系统

适用场景：
- 大型C++/Python/JavaScript等代码库
- 本地离线部署（RTX 3090 + Qwen模型）
- AI辅助代码维护和理解
"""

import os
import sys
from pathlib import Path

# 确保可以导入src模块
sys.path.insert(0, str(Path(__file__).parent / "src"))


def example_basic_usage():
    """
    示例1: 基础使用流程
    """
    print("=" * 60)
    print("示例1: 基础使用 - 从零开始构建代码RAG")
    print("=" * 60)
    
    from pipeline import create_pipeline
    
    # 步骤1: 创建RAG流水线
    # 可以通过配置文件或直接传参
    pipeline = create_pipeline(
        project_name="my-cpp-project",
        project_root="/path/to/your/codebase",  # 替换为你的代码路径
        
        # 分块配置
        chunk_max_tokens=1024,
        chunk_overlap_tokens=128,
        
        # Embedding配置 - 本地运行
        embedding_model="BAAI/bge-large-zh-v1.5",
        embedding_device="cuda",
        
        # 向量库配置
        vector_db_path="./data/chroma_db",
        
        # 检索配置
        retrieval_top_k=10,
        use_rerank=True,
        rerank_model="BAAI/bge-reranker-base",
        
        # LLM配置 - 连接本地vLLM服务
        llm_api_base="http://localhost:8000/v1",
        llm_model="Qwen/Qwen2.5-Coder-7B-Instruct",
    )
    
    # 步骤2: 构建索引（首次运行需要）
    # pipeline.build_index()
    
    # 步骤3: 搜索代码
    # results = pipeline.search("排序算法实现")
    # for r in results:
    #     print(f"[{r['score']:.3f}] {r['metadata']['file_path']}")
    
    # 步骤4: 问答
    # answer = pipeline.ask("这个项目中有哪些数据结构的实现？")
    # print(answer)


def example_from_config():
    """
    示例2: 从配置文件加载
    """
    print("\n" + "=" * 60)
    print("示例2: 从配置文件加载")
    print("=" * 60)
    
    from pipeline import CodeRAGPipeline
    
    # 使用config.yaml配置文件
    # pipeline = CodeRAGPipeline(config_path="config.yaml")
    
    # 交互式会话
    # pipeline.interactive_session()


def example_step_by_step():
    """
    示例3: 分步骤使用各组件
    """
    print("\n" + "=" * 60)
    print("示例3: 分步骤使用各组件（更精细控制）")
    print("=" * 60)
    
    # ---------- 步骤1: 代码分块 ----------
    print("\n[Step 1] 代码分块...")
    from chunker import CodeChunker
    
    chunker = CodeChunker(
        max_tokens=1024,
        overlap_tokens=128,
        include_context=True
    )
    
    # 分块单个文件
    # for chunk in chunker.chunk_file("example.cpp"):
    #     print(f"  [{chunk.chunk_type}] {chunk.symbol_name}")
    
    # 分块整个目录
    # chunks = list(chunker.chunk_directory(
    #     "/path/to/code",
    #     extensions=[".cpp", ".h"],
    #     exclude_patterns=["**/build/**"]
    # ))
    # print(f"  Generated {len(chunks)} chunks")
    
    # ---------- 步骤2: 生成Embedding ----------
    print("\n[Step 2] 生成Embedding...")
    from embedding import CodeEmbeddingManager, EmbeddingConfig
    
    embedding_config = EmbeddingConfig(
        model_name="BAAI/bge-large-zh-v1.5",
        device="cuda",
        batch_size=32,
        normalize=True
    )
    embedding_manager = CodeEmbeddingManager(embedding_config)
    
    # 编码代码块
    # embeddings = embedding_manager.encode_code(chunks)
    # print(f"  Embedding shape: {embeddings.shape}")
    
    # ---------- 步骤3: 存储到向量库 ----------
    print("\n[Step 3] 存储到向量库...")
    from vector_store import CodeVectorStore, VectorStoreConfig
    
    store_config = VectorStoreConfig(
        persist_directory="./data/chroma_db",
        collection_name="code_chunks"
    )
    vector_store = CodeVectorStore(store_config)
    
    # 添加到向量库
    # vector_store.add_chunks(chunks, embeddings)
    # print(f"  Total documents: {vector_store.count()}")
    
    # ---------- 步骤4: 创建检索器 ----------
    print("\n[Step 4] 创建检索器...")
    from retriever import CodeRetriever, RetrieverConfig
    
    retriever_config = RetrieverConfig(
        top_k=10,
        rerank=True,
        rerank_model="BAAI/bge-reranker-base",
        rerank_top_k=5,
        hybrid_search=True,
        semantic_weight=0.7,
        keyword_weight=0.3
    )
    
    retriever = CodeRetriever(
        embedding_manager,
        vector_store,
        retriever_config
    )
    
    # 检索
    # results = retriever.retrieve("二叉树遍历")
    # for r in results:
    #     print(f"  [{r['score']:.3f}] {r['metadata']['symbol_name']}")
    
    # ---------- 步骤5: 连接LLM ----------
    print("\n[Step 5] 连接LLM...")
    from llm import LLMManager, LLMConfig
    
    llm_config = LLMConfig(
        api_base="http://localhost:8000/v1",
        model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
        max_tokens=2048,
        temperature=0.1
    )
    llm_manager = LLMManager(llm_config, backend="openai")
    
    # 问答
    # answer = llm_manager.answer_code_question(
    #     "这段代码是做什么的？",
    #     results  # 检索到的上下文
    # )
    # print(answer)


def example_code_graph():
    """
    示例4: 使用代码图谱增强检索
    """
    print("\n" + "=" * 60)
    print("示例4: 代码图谱增强检索")
    print("=" * 60)
    
    from code_graph import CodeGraphBuilder, GraphEnhancedRetriever
    
    # 构建代码图
    builder = CodeGraphBuilder()
    
    # graph = builder.build_graph(
    #     "/path/to/code",
    #     extensions=[".cpp", ".h"],
    #     exclude_patterns=["**/build/**"]
    # )
    
    # 保存/加载图
    # builder.save_graph("./data/code_graph.json")
    # builder.load_graph("./data/code_graph.json")
    
    # 查询相关符号
    # related = builder.get_related_symbols("MyClass::process", max_depth=2)
    # print(f"Related symbols: {related}")
    
    # 获取调用链
    # chains = builder.get_call_chain("main", "Worker::run")
    # for chain in chains:
    #     print(" -> ".join(chain))
    
    # 图增强检索
    # enhanced_retriever = GraphEnhancedRetriever(base_retriever, builder)
    # results = enhanced_retriever.retrieve("数据处理", expand_relations=True)


def example_vllm_setup():
    """
    示例5: vLLM本地部署（RTX 3090推荐配置）
    """
    print("\n" + "=" * 60)
    print("示例5: vLLM本地部署指南")
    print("=" * 60)
    
    print("""
    # 1. 安装vLLM
    pip install vllm
    
    # 2. 启动vLLM服务（RTX 3090 24GB配置）
    python -m vllm.entrypoints.openai.api_server \\
        --model Qwen/Qwen2.5-Coder-7B-Instruct \\
        --host 0.0.0.0 \\
        --port 8000 \\
        --tensor-parallel-size 1 \\
        --gpu-memory-utilization 0.9 \\
        --max-model-len 8192
    
    # 3. 或使用Ollama（更简单）
    # 安装: https://ollama.ai
    # 拉取模型: ollama pull qwen2.5-coder:7b
    # 启动后自动在 http://localhost:11434
    
    # 4. 修改配置
    llm_config = LLMConfig(
        api_base="http://localhost:8000/v1",  # vLLM
        # api_base="http://localhost:11434",  # Ollama
        model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
    )
    """)


def example_cli_commands():
    """
    示例6: 命令行使用
    """
    print("\n" + "=" * 60)
    print("示例6: 命令行工具使用")
    print("=" * 60)
    
    print("""
    # 构建索引
    python src/cli.py index /path/to/code --name my-project
    
    # 搜索代码
    python src/cli.py search "排序算法" -k 10 --show-code
    
    # 问答
    python src/cli.py ask "这个项目的主要功能是什么？"
    
    # 解释代码
    python src/cli.py explain /path/to/file.cpp -s 100 -e 150
    
    # 交互式会话
    python src/cli.py chat
    
    # 查看统计
    python src/cli.py stats
    
    # 构建代码图谱
    python src/cli.py graph build -p /path/to/code -o graph.json
    """)


def example_best_practices():
    """
    示例7: 最佳实践
    """
    print("\n" + "=" * 60)
    print("示例7: 大型项目RAG最佳实践")
    print("=" * 60)
    
    print("""
    1. 分块策略
       - 使用语义分块（按函数/类）而非固定大小
       - 保留上下文信息（文件路径、符号名称）
       - 适当的overlap避免信息断裂
    
    2. Embedding选择
       - 中文项目: BAAI/bge-large-zh-v1.5
       - 代码专用: microsoft/codebert-base
       - 多语言: intfloat/multilingual-e5-large
    
    3. 检索优化
       - 启用混合检索（语义+关键词）
       - 使用Reranker精排
       - 考虑HyDE提升查询质量
    
    4. 上下文管理
       - 控制输入LLM的上下文长度
       - 按相关性排序后截断
       - 保留最相关的代码片段
    
    5. 增量更新
       - 跟踪文件修改时间
       - 只重新索引变化的文件
       - 定期优化向量库
    
    6. 代码图谱增强
       - 构建调用关系图
       - 检索时扩展相关符号
       - 理解代码结构层次
    """)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("   Code RAG System - 完整使用示例")
    print("   为大型项目构建AI代码理解系统")
    print("=" * 60)
    
    # 运行所有示例
    example_basic_usage()
    example_from_config()
    example_step_by_step()
    example_code_graph()
    example_vllm_setup()
    example_cli_commands()
    example_best_practices()
    
    print("\n" + "=" * 60)
    print("完成！请根据你的项目需求修改配置并运行。")
    print("=" * 60)
