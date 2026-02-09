"""
Code RAG System - Quick Start with Ollama
使用Ollama作为LLM后端的快速启动脚本
"""

import sys
import os

# 设置离线模式以避免网络请求
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
# os.environ["HF_HUB_OFFLINE"] = "1"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.pipeline import CodeRAGPipeline, create_pipeline
from src.llm import LLMManager, LLMConfig
import yaml

def load_config(config_path: str):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    print("正在启动 Code RAG System with Ollama...")
    
    # 加载Ollama配置
    config = load_config('config-ollama.yaml')
    
    # 创建LLM管理器，使用Ollama后端
    llm_config = LLMConfig(
        api_base=config['llm']['api_base'],
        model_name=config['llm']['model_name'],
        max_tokens=config['llm']['max_tokens'],
        temperature=config['llm']['temperature']
    )
    
    llm_manager = LLMManager(llm_config, backend="ollama")
    
    print(f"LLM后端: {config['llm']['backend']}")
    print(f"模型: {config['llm']['model_name']}")
    print(f"API地址: {config['llm']['api_base']}")
    
    # 创建RAG管道
    pipeline = create_pipeline(
        project_root=config['project']['root_path'],
        file_extensions=config['project']['file_extensions'],
        exclude_patterns=config['project']['exclude_patterns'],
        chunk_max_tokens=config['chunking']['max_tokens'],
        chunk_overlap_tokens=config['chunking']['overlap_tokens'],
        embedding_model=config['embedding']['model_name'],
        embedding_device=config['embedding']['device'],
        vector_db_path=config['vector_db']['persist_directory'],
        collection_name=config['vector_db']['collection_name'],
        retrieval_top_k=config['retrieval']['top_k'],
        use_rerank=config['retrieval']['rerank'],
        rerank_model=config['retrieval'].get('rerank_model'),
        rerank_top_k=config['retrieval'].get('rerank_top_k'),
        llm_api_base=config['llm']['api_base'],
        llm_model=config['llm']['model_name'],
        llm_max_tokens=config['llm']['max_tokens'],
        llm_temperature=config['llm']['temperature'],
        use_hybrid_search=config['retrieval']['hybrid_search'].get('enabled', True)
    )
    
    print("\n系统初始化完成！")
    print("请确保Ollama服务正在运行，并且已拉取所需模型。")
    print("启动Ollama服务命令: ollama serve")
    print("拉取模型命令: ollama pull qwen2.5-coder:14b-instruct-q8_0")
    
    # 进入交互模式
    print("\n输入 'quit' 退出程序")
    while True:
        try:
            question = input("\n请输入您的代码相关问题: ")
            if question.lower().strip() == 'quit':
                break
            
            print("\n正在检索相关信息并生成答案...")
            
            # 获取答案
            response = pipeline.query(question)
            print(f"\n答案:\n{response}")
            
        except KeyboardInterrupt:
            print("\n\n程序被用户中断")
            break
        except Exception as e:
            print(f"\n发生错误: {str(e)}")
            continue
    
    print("感谢使用 Code RAG System!")

if __name__ == "__main__":
    main()