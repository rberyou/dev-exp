#!/usr/bin/env python3
"""
Code RAG System - 命令行工具
"""

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent))

from pipeline import CodeRAGPipeline, RAGConfig, create_pipeline


console = Console()


def cmd_index(args):
    """构建索引命令"""
    pipeline = create_pipeline(
        config_path=args.config,
        project_root=args.path,
        project_name=args.name or Path(args.path).name
    )
    
    pipeline.build_index(
        root_path=args.path,
        incremental=not args.rebuild
    )


def cmd_search(args):
    """搜索命令"""
    pipeline = create_pipeline(config_path=args.config)
    
    results = pipeline.search(
        args.query,
        top_k=args.top_k,
        file_filter=args.file,
        language_filter=args.language
    )
    
    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return
    
    for i, result in enumerate(results, 1):
        meta = result.get('metadata', {})
        score = result.get('rerank_score', result.get('score', 0))
        
        console.print(f"\n[bold cyan]#{i}[/bold cyan] [{score:.3f}] {meta.get('file_path', '?')}")
        console.print(f"    Type: {meta.get('chunk_type', '?')} | Symbol: {meta.get('symbol_name', '-')}")
        console.print(f"    Lines: {meta.get('start_line', '?')}-{meta.get('end_line', '?')}")
        
        if args.show_code:
            console.print(Panel(result.get('content', '')[:500], title="Code Preview"))


def cmd_ask(args):
    """问答命令"""
    pipeline = create_pipeline(config_path=args.config)
    
    console.print(f"\n[bold blue]Question:[/bold blue] {args.question}\n")
    console.print("[bold green]Answer:[/bold green]")
    
    if args.stream:
        for chunk in pipeline.ask(args.question, stream=True, show_context=args.show_context):
            console.print(chunk, end="")
        console.print()
    else:
        answer = pipeline.ask(args.question, show_context=args.show_context)
        console.print(answer)


def cmd_explain(args):
    """代码解释命令"""
    pipeline = create_pipeline(config_path=args.config)
    
    console.print(f"\n[bold]Explaining:[/bold] {args.file}")
    if args.start_line and args.end_line:
        console.print(f"Lines: {args.start_line}-{args.end_line}")
    
    console.print("\n[bold green]Explanation:[/bold green]")
    
    for chunk in pipeline.explain_code(
        args.file,
        start_line=args.start_line,
        end_line=args.end_line,
        stream=True
    ):
        console.print(chunk, end="")
    console.print()


def cmd_interactive(args):
    """交互式会话"""
    pipeline = create_pipeline(config_path=args.config)
    pipeline.interactive_session()


def cmd_stats(args):
    """显示统计信息"""
    pipeline = create_pipeline(config_path=args.config)
    stats = pipeline.get_stats()
    console.print_json(data=stats)


def cmd_graph(args):
    """代码图谱命令"""
    from code_graph import CodeGraphBuilder
    
    builder = CodeGraphBuilder()
    
    if args.action == 'build':
        graph = builder.build_graph(
            args.path,
            extensions=args.extensions.split(',') if args.extensions else None
        )
        if args.output:
            builder.save_graph(args.output)
        console.print(f"[green]Graph built: {len(builder.symbols)} symbols[/green]")
    
    elif args.action == 'query':
        builder.load_graph(args.input)
        related = builder.get_related_symbols(args.symbol, max_depth=args.depth)
        console.print(f"[bold]Related to {args.symbol}:[/bold]")
        for sym in related:
            console.print(f"  - {sym}")


def main():
    parser = argparse.ArgumentParser(
        description='Code RAG System - AI-powered code search and understanding',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('-c', '--config', default='config.yaml', help='Configuration file path')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # index 命令
    index_parser = subparsers.add_parser('index', help='Build code index')
    index_parser.add_argument('path', help='Code directory path')
    index_parser.add_argument('-n', '--name', help='Project name')
    index_parser.add_argument('--rebuild', action='store_true', help='Rebuild index from scratch')
    index_parser.set_defaults(func=cmd_index)
    
    # search 命令
    search_parser = subparsers.add_parser('search', help='Search code')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('-k', '--top-k', type=int, default=5, help='Number of results')
    search_parser.add_argument('-f', '--file', help='Filter by file path prefix')
    search_parser.add_argument('-l', '--language', help='Filter by language')
    search_parser.add_argument('--show-code', action='store_true', help='Show code preview')
    search_parser.set_defaults(func=cmd_search)
    
    # ask 命令
    ask_parser = subparsers.add_parser('ask', help='Ask a question about the code')
    ask_parser.add_argument('question', help='Your question')
    ask_parser.add_argument('--stream', action='store_true', default=True, help='Stream output')
    ask_parser.add_argument('--show-context', action='store_true', help='Show retrieved context')
    ask_parser.set_defaults(func=cmd_ask)
    
    # explain 命令
    explain_parser = subparsers.add_parser('explain', help='Explain code')
    explain_parser.add_argument('file', help='File path')
    explain_parser.add_argument('-s', '--start-line', type=int, help='Start line')
    explain_parser.add_argument('-e', '--end-line', type=int, help='End line')
    explain_parser.set_defaults(func=cmd_explain)
    
    # interactive 命令
    interactive_parser = subparsers.add_parser('chat', help='Start interactive session')
    interactive_parser.set_defaults(func=cmd_interactive)
    
    # stats 命令
    stats_parser = subparsers.add_parser('stats', help='Show system statistics')
    stats_parser.set_defaults(func=cmd_stats)
    
    # graph 命令
    graph_parser = subparsers.add_parser('graph', help='Code graph operations')
    graph_parser.add_argument('action', choices=['build', 'query'], help='Action')
    graph_parser.add_argument('-p', '--path', help='Code directory (for build)')
    graph_parser.add_argument('-o', '--output', help='Output file (for build)')
    graph_parser.add_argument('-i', '--input', help='Input graph file (for query)')
    graph_parser.add_argument('-s', '--symbol', help='Symbol to query')
    graph_parser.add_argument('-d', '--depth', type=int, default=2, help='Search depth')
    graph_parser.add_argument('--extensions', help='File extensions (comma-separated)')
    graph_parser.set_defaults(func=cmd_graph)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        args.func(args)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if '--debug' in sys.argv:
            raise


if __name__ == '__main__':
    main()
