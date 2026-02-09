#!/usr/bin/env python3
"""
快速启动脚本 - 一键构建代码RAG

使用方法:
    python quickstart.py /path/to/your/codebase

依赖安装:
    pip install -r requirements.txt
"""

import sys
import os
from pathlib import Path

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))


def main():
    if len(sys.argv) < 2:
        print("使用方法: python quickstart.py <代码目录路径>")
        print("示例: python quickstart.py /home/user/my-project")
        sys.exit(1)
    
    code_path = sys.argv[1]
    
    if not os.path.isdir(code_path):
        print(f"错误: 目录不存在 - {code_path}")
        sys.exit(1)
    
    project_name = Path(code_path).name
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║           Code RAG System - 快速启动                          ║
╠══════════════════════════════════════════════════════════════╣
║  项目: {project_name:<52} ║
║  路径: {code_path:<52} ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    # 检查LLM服务
    print("[1/4] 检查LLM服务...")
    import requests
    llm_available = False
    try:
        resp = requests.get("http://localhost:8000/v1/models", timeout=2)
        if resp.status_code == 200:
            llm_available = True
            print("  ✓ vLLM服务已启动")
    except:
        pass
    
    if not llm_available:
        try:
            resp = requests.get("http://localhost:11434/api/tags", timeout=2)
            if resp.status_code == 200:
                llm_available = True
                print("  ✓ Ollama服务已启动")
        except:
            pass
    
    if not llm_available:
        print("  ⚠ 未检测到LLM服务，问答功能将不可用")
        print("    请启动 vLLM 或 Ollama 服务")
    
    # 创建流水线
    print("\n[2/4] 初始化RAG系统...")
    from pipeline import create_pipeline
    
    # 根据可用的服务确定API基础URL
    if llm_available:
        # 检测具体哪个服务可用
        import requests
        try:
            resp = requests.get("http://localhost:8000/v1/models", timeout=2)
            if resp.status_code == 200:
                llm_api_base = "http://localhost:8000/v1"
                print("  使用 vLLM 服务")
        except:
            try:
                resp = requests.get("http://localhost:11434/api/tags", timeout=2)
                if resp.status_code == 200:
                    llm_api_base = "http://localhost:11434/v1"
                    print("  使用 Ollama 服务")
            except:
                llm_api_base = None
                print("  ⚠ 未检测到有效的LLM服务")
    else:
        llm_api_base = None
    
    pipeline = create_pipeline(
        project_name=project_name,
        project_root=code_path,
        vector_db_path=f"./data/{project_name}_db",
        llm_api_base=llm_api_base,
    )
    
    # 构建索引
    print("\n[3/4] 构建代码索引...")
    stats = pipeline.build_index()
    
    # 启动交互
    print("\n[4/4] 启动交互式会话...")
    print("\n" + "=" * 60)
    
    if llm_available:
        pipeline.interactive_session()
    else:
        print("LLM服务未启动，仅支持搜索功能")
        print("输入查询进行代码搜索，输入 'quit' 退出\n")
        
        while True:
            query = input("搜索> ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                break
            if not query:
                continue
            
            results = pipeline.search(query, top_k=5)
            for i, r in enumerate(results, 1):
                meta = r.get('metadata', {})
                print(f"\n#{i} [{r.get('score', 0):.3f}] {meta.get('file_path', '?')}")
                print(f"   {meta.get('chunk_type', '?')} | {meta.get('symbol_name', '-')}")


if __name__ == "__main__":
    main()
