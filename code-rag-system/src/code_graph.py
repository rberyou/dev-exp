"""
Code RAG System - 代码关系图分析器
构建代码调用关系、继承关系等图谱，增强检索效果
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import json

import networkx as nx
import tree_sitter_languages


@dataclass
class CodeSymbol:
    """代码符号"""
    name: str
    qualified_name: str  # 完整限定名
    symbol_type: str     # function, class, method, variable, etc.
    file_path: str
    start_line: int
    end_line: int
    language: str
    parent: Optional[str] = None  # 父符号
    signature: str = ""           # 函数签名等
    docstring: str = ""           # 文档字符串
    metadata: Dict = field(default_factory=dict)


@dataclass
class CodeRelation:
    """代码关系"""
    source: str          # 源符号
    target: str          # 目标符号
    relation_type: str   # calls, inherits, implements, includes, etc.
    file_path: str
    line_number: int


class CodeGraphBuilder:
    """
    代码关系图构建器
    解析代码，提取符号和关系
    """
    
    # 语言配置
    LANG_CONFIG = {
        'cpp': {
            'class_nodes': ['class_specifier', 'struct_specifier'],
            'function_nodes': ['function_definition', 'function_declarator'],
            'call_nodes': ['call_expression'],
            'include_nodes': ['preproc_include'],
            'inheritance_nodes': ['base_class_clause'],
        },
        'python': {
            'class_nodes': ['class_definition'],
            'function_nodes': ['function_definition'],
            'call_nodes': ['call'],
            'import_nodes': ['import_statement', 'import_from_statement'],
            'inheritance_nodes': ['argument_list'],  # class Foo(Base):
        },
        'javascript': {
            'class_nodes': ['class_declaration'],
            'function_nodes': ['function_declaration', 'method_definition', 'arrow_function'],
            'call_nodes': ['call_expression'],
            'import_nodes': ['import_statement'],
            'inheritance_nodes': ['class_heritage'],
        },
    }
    
    def __init__(self):
        self.symbols: Dict[str, CodeSymbol] = {}
        self.relations: List[CodeRelation] = []
        self.graph = nx.DiGraph()
    
    def _get_language(self, file_path: str) -> Optional[str]:
        """获取文件语言"""
        ext = Path(file_path).suffix.lower()
        lang_map = {
            '.cpp': 'cpp', '.hpp': 'cpp', '.cc': 'cpp', '.h': 'cpp', '.c': 'cpp',
            '.py': 'python',
            '.js': 'javascript', '.jsx': 'javascript',
            '.ts': 'typescript', '.tsx': 'typescript',
        }
        return lang_map.get(ext)
    
    def _extract_name(self, node, source: bytes, language: str) -> Optional[str]:
        """提取节点名称"""
        for child in node.children:
            if child.type in ['identifier', 'name', 'type_identifier', 'property_identifier']:
                return source[child.start_byte:child.end_byte].decode('utf-8', errors='ignore')
            if child.type in ['declarator', 'function_declarator']:
                return self._extract_name(child, source, language)
        return None
    
    def _build_qualified_name(self, name: str, parent: str = None, file_path: str = "") -> str:
        """构建完整限定名"""
        parts = []
        if file_path:
            # 使用相对路径作为模块名
            parts.append(Path(file_path).stem)
        if parent:
            parts.append(parent)
        parts.append(name)
        return "::".join(parts)
    
    def parse_file(self, file_path: str) -> Tuple[List[CodeSymbol], List[CodeRelation]]:
        """
        解析单个文件
        
        Returns:
            (符号列表, 关系列表)
        """
        language = self._get_language(file_path)
        if not language or language not in self.LANG_CONFIG:
            return [], []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                source = f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return [], []
        
        source_bytes = source.encode('utf-8')
        
        try:
            parser = tree_sitter_languages.get_parser(language)
            tree = parser.parse(source_bytes)
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return [], []
        
        config = self.LANG_CONFIG[language]
        symbols = []
        relations = []
        
        # 遍历语法树
        def traverse(node, parent_symbol=None):
            nonlocal symbols, relations
            
            # 提取类
            if node.type in config.get('class_nodes', []):
                name = self._extract_name(node, source_bytes, language)
                if name:
                    qualified_name = self._build_qualified_name(name, None, file_path)
                    symbol = CodeSymbol(
                        name=name,
                        qualified_name=qualified_name,
                        symbol_type='class',
                        file_path=file_path,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        language=language,
                    )
                    symbols.append(symbol)
                    
                    # 提取继承关系
                    for child in node.children:
                        if child.type in config.get('inheritance_nodes', []):
                            base_names = self._extract_base_classes(child, source_bytes, language)
                            for base_name in base_names:
                                relations.append(CodeRelation(
                                    source=qualified_name,
                                    target=base_name,
                                    relation_type='inherits',
                                    file_path=file_path,
                                    line_number=node.start_point[0] + 1
                                ))
                    
                    # 递归处理类内部
                    for child in node.children:
                        traverse(child, name)
                    return
            
            # 提取函数
            if node.type in config.get('function_nodes', []):
                name = self._extract_name(node, source_bytes, language)
                if name:
                    qualified_name = self._build_qualified_name(name, parent_symbol, file_path)
                    symbol = CodeSymbol(
                        name=name,
                        qualified_name=qualified_name,
                        symbol_type='method' if parent_symbol else 'function',
                        file_path=file_path,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        language=language,
                        parent=parent_symbol,
                    )
                    symbols.append(symbol)
                    
                    # 提取函数调用
                    self._extract_calls(node, source_bytes, language, qualified_name, file_path, relations)
            
            # 提取include/import
            if node.type in config.get('include_nodes', []) + config.get('import_nodes', []):
                include_name = source_bytes[node.start_byte:node.end_byte].decode('utf-8', errors='ignore')
                # 简单记录依赖关系
                relations.append(CodeRelation(
                    source=file_path,
                    target=include_name,
                    relation_type='includes',
                    file_path=file_path,
                    line_number=node.start_point[0] + 1
                ))
            
            # 递归子节点
            for child in node.children:
                traverse(child, parent_symbol)
        
        traverse(tree.root_node)
        return symbols, relations
    
    def _extract_base_classes(self, node, source: bytes, language: str) -> List[str]:
        """提取基类名称"""
        base_classes = []
        
        def find_names(n):
            if n.type in ['identifier', 'type_identifier', 'scoped_identifier']:
                base_classes.append(source[n.start_byte:n.end_byte].decode('utf-8', errors='ignore'))
            for child in n.children:
                find_names(child)
        
        find_names(node)
        return base_classes
    
    def _extract_calls(
        self,
        node,
        source: bytes,
        language: str,
        caller: str,
        file_path: str,
        relations: List[CodeRelation]
    ):
        """提取函数调用关系"""
        config = self.LANG_CONFIG.get(language, {})
        
        def find_calls(n):
            if n.type in config.get('call_nodes', []):
                # 提取被调用的函数名
                callee = None
                for child in n.children:
                    if child.type in ['identifier', 'field_expression', 'scoped_identifier']:
                        callee = source[child.start_byte:child.end_byte].decode('utf-8', errors='ignore')
                        break
                
                if callee:
                    relations.append(CodeRelation(
                        source=caller,
                        target=callee,
                        relation_type='calls',
                        file_path=file_path,
                        line_number=n.start_point[0] + 1
                    ))
            
            for child in n.children:
                find_calls(child)
        
        find_calls(node)
    
    def build_graph(
        self,
        root_path: str,
        extensions: List[str] = None,
        exclude_patterns: List[str] = None
    ) -> nx.DiGraph:
        """
        构建代码关系图
        
        Args:
            root_path: 代码根目录
            extensions: 文件扩展名列表
            exclude_patterns: 排除模式
        
        Returns:
            NetworkX有向图
        """
        import fnmatch
        
        if extensions is None:
            extensions = ['.cpp', '.h', '.hpp', '.py', '.js', '.ts']
        if exclude_patterns is None:
            exclude_patterns = ['**/build/**', '**/node_modules/**', '**/.git/**']
        
        root = Path(root_path)
        
        all_symbols = []
        all_relations = []
        
        # 遍历所有文件
        for file_path in root.rglob('*'):
            if not file_path.is_file():
                continue
            
            if file_path.suffix.lower() not in extensions:
                continue
            
            rel_path = str(file_path.relative_to(root))
            if any(fnmatch.fnmatch(rel_path, p) for p in exclude_patterns):
                continue
            
            symbols, relations = self.parse_file(str(file_path))
            all_symbols.extend(symbols)
            all_relations.extend(relations)
        
        # 存储符号
        for symbol in all_symbols:
            self.symbols[symbol.qualified_name] = symbol
            self.graph.add_node(
                symbol.qualified_name,
                **{
                    'name': symbol.name,
                    'type': symbol.symbol_type,
                    'file': symbol.file_path,
                    'start_line': symbol.start_line,
                    'end_line': symbol.end_line,
                }
            )
        
        # 存储关系
        for relation in all_relations:
            self.relations.append(relation)
            self.graph.add_edge(
                relation.source,
                relation.target,
                relation_type=relation.relation_type,
                file=relation.file_path,
                line=relation.line_number
            )
        
        print(f"Graph built: {len(self.symbols)} symbols, {len(self.relations)} relations")
        
        return self.graph
    
    def get_related_symbols(
        self,
        symbol_name: str,
        relation_types: List[str] = None,
        max_depth: int = 2
    ) -> List[str]:
        """
        获取相关符号
        
        Args:
            symbol_name: 符号名称
            relation_types: 关系类型过滤
            max_depth: 最大深度
        
        Returns:
            相关符号列表
        """
        if symbol_name not in self.graph:
            # 尝试模糊匹配
            matches = [n for n in self.graph.nodes if symbol_name in n]
            if matches:
                symbol_name = matches[0]
            else:
                return []
        
        related = set()
        
        # BFS遍历
        queue = [(symbol_name, 0)]
        visited = {symbol_name}
        
        while queue:
            current, depth = queue.pop(0)
            
            if depth >= max_depth:
                continue
            
            # 获取出边（调用的函数）
            for _, target, data in self.graph.out_edges(current, data=True):
                if relation_types and data.get('relation_type') not in relation_types:
                    continue
                related.add(target)
                if target not in visited:
                    visited.add(target)
                    queue.append((target, depth + 1))
            
            # 获取入边（被调用）
            for source, _, data in self.graph.in_edges(current, data=True):
                if relation_types and data.get('relation_type') not in relation_types:
                    continue
                related.add(source)
                if source not in visited:
                    visited.add(source)
                    queue.append((source, depth + 1))
        
        return list(related)
    
    def get_call_chain(self, from_symbol: str, to_symbol: str) -> List[List[str]]:
        """获取调用链"""
        try:
            paths = list(nx.all_simple_paths(self.graph, from_symbol, to_symbol, cutoff=5))
            return paths
        except nx.NetworkXError:
            return []
    
    def get_class_hierarchy(self, class_name: str) -> Dict:
        """获取类继承层次"""
        hierarchy = {
            'name': class_name,
            'bases': [],
            'derived': []
        }
        
        # 查找基类
        for _, target, data in self.graph.out_edges(class_name, data=True):
            if data.get('relation_type') == 'inherits':
                hierarchy['bases'].append(target)
        
        # 查找派生类
        for source, _, data in self.graph.in_edges(class_name, data=True):
            if data.get('relation_type') == 'inherits':
                hierarchy['derived'].append(source)
        
        return hierarchy
    
    def save_graph(self, output_path: str):
        """保存图到文件"""
        data = {
            'symbols': {k: {
                'name': v.name,
                'qualified_name': v.qualified_name,
                'type': v.symbol_type,
                'file': v.file_path,
                'start_line': v.start_line,
                'end_line': v.end_line,
            } for k, v in self.symbols.items()},
            'relations': [{
                'source': r.source,
                'target': r.target,
                'type': r.relation_type,
                'file': r.file_path,
                'line': r.line_number,
            } for r in self.relations]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Graph saved to {output_path}")
    
    def load_graph(self, input_path: str):
        """从文件加载图"""
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 重建符号
        for name, info in data['symbols'].items():
            symbol = CodeSymbol(
                name=info['name'],
                qualified_name=info['qualified_name'],
                symbol_type=info['type'],
                file_path=info['file'],
                start_line=info['start_line'],
                end_line=info['end_line'],
                language='unknown'
            )
            self.symbols[name] = symbol
            self.graph.add_node(name, **info)
        
        # 重建关系
        for rel in data['relations']:
            relation = CodeRelation(
                source=rel['source'],
                target=rel['target'],
                relation_type=rel['type'],
                file_path=rel['file'],
                line_number=rel['line']
            )
            self.relations.append(relation)
            self.graph.add_edge(rel['source'], rel['target'], **rel)
        
        print(f"Graph loaded: {len(self.symbols)} symbols, {len(self.relations)} relations")


class GraphEnhancedRetriever:
    """
    图增强检索器
    结合向量检索和代码图谱进行检索
    """
    
    def __init__(self, base_retriever, code_graph: CodeGraphBuilder):
        self.base_retriever = base_retriever
        self.code_graph = code_graph
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        expand_relations: bool = True,
        relation_types: List[str] = None,
        **kwargs
    ) -> List[Dict]:
        """
        图增强检索
        
        Args:
            query: 查询
            top_k: 返回数量
            expand_relations: 是否扩展相关符号
            relation_types: 要扩展的关系类型
        
        Returns:
            检索结果
        """
        # 基础检索
        results = self.base_retriever.retrieve(query, top_k=top_k, **kwargs)
        
        if not expand_relations or not results:
            return results
        
        # 提取检索到的符号
        symbols_found = set()
        for result in results:
            symbol_name = result.get('metadata', {}).get('symbol_name')
            if symbol_name:
                symbols_found.add(symbol_name)
        
        # 扩展相关符号
        related_symbols = set()
        for symbol in symbols_found:
            related = self.code_graph.get_related_symbols(
                symbol,
                relation_types=relation_types,
                max_depth=1
            )
            related_symbols.update(related)
        
        # 如果有新的相关符号，补充检索
        new_symbols = related_symbols - symbols_found
        if new_symbols:
            # 从图中获取这些符号的信息
            additional_results = []
            for symbol_name in list(new_symbols)[:5]:  # 限制数量
                if symbol_name in self.code_graph.symbols:
                    symbol = self.code_graph.symbols[symbol_name]
                    additional_results.append({
                        'id': f"graph_{symbol_name}",
                        'content': f"[From code graph] {symbol_name}",
                        'metadata': {
                            'file_path': symbol.file_path,
                            'symbol_name': symbol.name,
                            'chunk_type': symbol.symbol_type,
                            'start_line': symbol.start_line,
                            'end_line': symbol.end_line,
                        },
                        'score': 0.5,  # 图扩展的结果给一个中等分数
                        'source': 'graph_expansion'
                    })
            
            results.extend(additional_results)
        
        return results[:top_k]


# 使用示例
if __name__ == "__main__":
    # 构建代码图
    builder = CodeGraphBuilder()
    graph = builder.build_graph(
        "/path/to/project",
        extensions=[".cpp", ".h"],
        exclude_patterns=["**/build/**"]
    )
    
    # 保存图
    # builder.save_graph("./data/code_graph.json")
    
    # 查询相关符号
    # related = builder.get_related_symbols("MyClass::myMethod", max_depth=2)
    # print(f"Related symbols: {related}")
    
    # 获取调用链
    # chains = builder.get_call_chain("main", "process")
    # for chain in chains:
    #     print(" -> ".join(chain))
