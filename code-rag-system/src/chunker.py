"""
Code RAG System - 代码智能分块器
使用 tree-sitter 进行语法感知的代码分块
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Generator
import tree_sitter_languages
from tree_sitter import Node
import tiktoken


@dataclass
class CodeChunk:
    """代码块数据结构"""
    id: str
    content: str
    file_path: str
    language: str
    start_line: int
    end_line: int
    chunk_type: str  # function, class, module, etc.
    symbol_name: Optional[str] = None
    parent_symbols: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    
    @property
    def context_header(self) -> str:
        """生成上下文头部信息"""
        header = f"// File: {self.file_path}\n"
        header += f"// Type: {self.chunk_type}"
        if self.symbol_name:
            header += f" | Name: {self.symbol_name}"
        header += f"\n// Lines: {self.start_line}-{self.end_line}\n"
        if self.parent_symbols:
            header += f"// Parent: {' > '.join(self.parent_symbols)}\n"
        return header
    
    @property
    def full_content(self) -> str:
        """包含上下文的完整内容"""
        return self.context_header + self.content


class CodeChunker:
    """智能代码分块器"""
    
    # 语言到文件扩展名的映射
    LANG_MAP = {
        '.cpp': 'cpp', '.hpp': 'cpp', '.cc': 'cpp', '.cxx': 'cpp',
        '.c': 'c', '.h': 'c',
        '.py': 'python',
        '.js': 'javascript', '.jsx': 'javascript',
        '.ts': 'typescript', '.tsx': 'typescript',
        '.java': 'java',
        '.go': 'go',
        '.rs': 'rust',
    }
    
    # 各语言的语义节点类型
    SEMANTIC_NODES = {
        'cpp': ['function_definition', 'class_specifier', 'struct_specifier', 
                'namespace_definition', 'template_declaration'],
        'c': ['function_definition', 'struct_specifier', 'enum_specifier'],
        'python': ['function_definition', 'class_definition', 'decorated_definition'],
        'javascript': ['function_declaration', 'class_declaration', 'method_definition',
                      'arrow_function', 'function_expression'],
        'typescript': ['function_declaration', 'class_declaration', 'method_definition',
                      'interface_declaration', 'type_alias_declaration'],
        'java': ['method_declaration', 'class_declaration', 'interface_declaration'],
        'go': ['function_declaration', 'method_declaration', 'type_declaration'],
        'rust': ['function_item', 'impl_item', 'struct_item', 'enum_item', 'trait_item'],
    }
    
    def __init__(
        self,
        max_tokens: int = 1024,
        overlap_tokens: int = 128,
        include_context: bool = True
    ):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.include_context = include_context
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self._chunk_counter = 0
    
    def _get_language(self, file_path: str) -> Optional[str]:
        """根据文件扩展名获取语言"""
        ext = Path(file_path).suffix.lower()
        return self.LANG_MAP.get(ext)
    
    def _count_tokens(self, text: str) -> int:
        """计算token数量"""
        return len(self.tokenizer.encode(text))
    
    def _extract_symbol_name(self, node: Node, source: bytes, language: str) -> Optional[str]:
        """提取符号名称"""
        name_nodes = {
            'cpp': ['declarator', 'name', 'identifier'],
            'c': ['declarator', 'identifier'],
            'python': ['name', 'identifier'],
            'javascript': ['name', 'identifier', 'property_identifier'],
            'typescript': ['name', 'identifier', 'type_identifier'],
            'java': ['name', 'identifier'],
            'go': ['name', 'identifier', 'field_identifier'],
            'rust': ['name', 'identifier'],
        }
        
        for child in node.children:
            if child.type in name_nodes.get(language, ['identifier', 'name']):
                return source[child.start_byte:child.end_byte].decode('utf-8', errors='ignore')
            # 递归查找
            if child.type in ['declarator', 'function_declarator']:
                result = self._extract_symbol_name(child, source, language)
                if result:
                    return result
        return None
    
    def _extract_imports(self, tree, source: bytes, language: str) -> List[str]:
        """提取导入语句"""
        imports = []
        import_types = {
            'cpp': ['preproc_include'],
            'c': ['preproc_include'],
            'python': ['import_statement', 'import_from_statement'],
            'javascript': ['import_statement'],
            'typescript': ['import_statement'],
            'java': ['import_declaration'],
            'go': ['import_declaration'],
            'rust': ['use_declaration'],
        }
        
        def traverse(node):
            if node.type in import_types.get(language, []):
                imports.append(source[node.start_byte:node.end_byte].decode('utf-8', errors='ignore'))
            for child in node.children:
                traverse(child)
        
        traverse(tree.root_node)
        return imports
    
    def _get_parent_symbols(self, node: Node, source: bytes, language: str) -> List[str]:
        """获取父级符号链"""
        parents = []
        current = node.parent
        semantic_nodes = self.SEMANTIC_NODES.get(language, [])
        
        while current:
            if current.type in semantic_nodes:
                name = self._extract_symbol_name(current, source, language)
                if name:
                    parents.insert(0, f"{current.type}:{name}")
            current = current.parent
        
        return parents
    
    def _generate_chunk_id(self, file_path: str) -> str:
        """生成唯一的chunk ID"""
        self._chunk_counter += 1
        file_hash = hash(file_path) % 10000
        return f"chunk_{file_hash}_{self._chunk_counter}"
    
    def _split_large_node(
        self,
        node: Node,
        source: bytes,
        file_path: str,
        language: str,
        imports: List[str],
        parent_symbols: List[str]
    ) -> Generator[CodeChunk, None, None]:
        """将过大的节点按行分割"""
        content = source[node.start_byte:node.end_byte].decode('utf-8', errors='ignore')
        lines = content.split('\n')
        
        current_chunk = []
        current_tokens = 0
        chunk_start_line = node.start_point[0] + 1
        
        for i, line in enumerate(lines):
            line_tokens = self._count_tokens(line + '\n')
            
            if current_tokens + line_tokens > self.max_tokens and current_chunk:
                # 输出当前chunk
                chunk_content = '\n'.join(current_chunk)
                yield CodeChunk(
                    id=self._generate_chunk_id(file_path),
                    content=chunk_content,
                    file_path=file_path,
                    language=language,
                    start_line=chunk_start_line,
                    end_line=chunk_start_line + len(current_chunk) - 1,
                    chunk_type=f"{node.type}_part",
                    symbol_name=self._extract_symbol_name(node, source, language),
                    parent_symbols=parent_symbols.copy(),
                    imports=imports.copy(),
                )
                
                # 保留overlap
                overlap_lines = []
                overlap_tokens = 0
                for prev_line in reversed(current_chunk):
                    prev_tokens = self._count_tokens(prev_line + '\n')
                    if overlap_tokens + prev_tokens <= self.overlap_tokens:
                        overlap_lines.insert(0, prev_line)
                        overlap_tokens += prev_tokens
                    else:
                        break
                
                current_chunk = overlap_lines
                current_tokens = overlap_tokens
                chunk_start_line = chunk_start_line + len(current_chunk) - len(overlap_lines)
            
            current_chunk.append(line)
            current_tokens += line_tokens
        
        # 输出剩余内容
        if current_chunk:
            chunk_content = '\n'.join(current_chunk)
            yield CodeChunk(
                id=self._generate_chunk_id(file_path),
                content=chunk_content,
                file_path=file_path,
                language=language,
                start_line=chunk_start_line,
                end_line=node.end_point[0] + 1,
                chunk_type=f"{node.type}_part",
                symbol_name=self._extract_symbol_name(node, source, language),
                parent_symbols=parent_symbols.copy(),
                imports=imports.copy(),
            )
    
    def chunk_file(self, file_path: str) -> Generator[CodeChunk, None, None]:
        """对单个文件进行分块"""
        language = self._get_language(file_path)
        if not language:
            return
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                source = f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return
        
        source_bytes = source.encode('utf-8')
        
        # 解析代码
        try:
            parser = tree_sitter_languages.get_parser(language)
            tree = parser.parse(source_bytes)
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            # 回退到简单分块
            yield from self._fallback_chunk(source, file_path, language)
            return
        
        # 提取imports
        imports = self._extract_imports(tree, source_bytes, language)
        
        # 遍历语法树，找到语义节点
        semantic_nodes = self.SEMANTIC_NODES.get(language, [])
        processed_ranges = set()
        
        def traverse(node: Node, depth: int = 0):
            if node.type in semantic_nodes:
                # 检查是否已处理
                node_range = (node.start_byte, node.end_byte)
                if node_range in processed_ranges:
                    return
                
                content = source_bytes[node.start_byte:node.end_byte].decode('utf-8', errors='ignore')
                token_count = self._count_tokens(content)
                parent_symbols = self._get_parent_symbols(node, source_bytes, language)
                
                if token_count <= self.max_tokens:
                    # 节点大小合适，直接输出
                    yield CodeChunk(
                        id=self._generate_chunk_id(file_path),
                        content=content,
                        file_path=file_path,
                        language=language,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        chunk_type=node.type,
                        symbol_name=self._extract_symbol_name(node, source_bytes, language),
                        parent_symbols=parent_symbols,
                        imports=imports.copy(),
                    )
                    processed_ranges.add(node_range)
                else:
                    # 节点过大，需要分割
                    yield from self._split_large_node(
                        node, source_bytes, file_path, language, imports, parent_symbols
                    )
                    processed_ranges.add(node_range)
            else:
                # 继续遍历子节点
                for child in node.children:
                    yield from traverse(child, depth + 1)
        
        yield from traverse(tree.root_node)
        
        # 处理未被语义节点覆盖的代码（如全局变量、宏定义等）
        yield from self._chunk_remaining(
            tree.root_node, source_bytes, file_path, language, imports, processed_ranges
        )
    
    def _chunk_remaining(
        self,
        root: Node,
        source: bytes,
        file_path: str,
        language: str,
        imports: List[str],
        processed_ranges: set
    ) -> Generator[CodeChunk, None, None]:
        """处理未被语义节点覆盖的代码"""
        # 找出未处理的范围
        total_length = len(source)
        processed_bytes = set()
        
        for start, end in processed_ranges:
            for i in range(start, end):
                processed_bytes.add(i)
        
        # 收集未处理的连续范围
        remaining = []
        current_start = None
        
        for i in range(total_length):
            if i not in processed_bytes:
                if current_start is None:
                    current_start = i
            else:
                if current_start is not None:
                    remaining.append((current_start, i))
                    current_start = None
        
        if current_start is not None:
            remaining.append((current_start, total_length))
        
        # 对未处理的范围进行分块
        for start, end in remaining:
            content = source[start:end].decode('utf-8', errors='ignore').strip()
            if not content or self._count_tokens(content) < 20:
                continue  # 跳过太短的内容
            
            # 计算行号
            start_line = source[:start].count(b'\n') + 1
            end_line = source[:end].count(b'\n') + 1
            
            token_count = self._count_tokens(content)
            
            if token_count <= self.max_tokens:
                yield CodeChunk(
                    id=self._generate_chunk_id(file_path),
                    content=content,
                    file_path=file_path,
                    language=language,
                    start_line=start_line,
                    end_line=end_line,
                    chunk_type="module_level",
                    imports=imports.copy(),
                )
            else:
                # 按行分割
                lines = content.split('\n')
                current_chunk = []
                current_tokens = 0
                chunk_start = start_line
                
                for i, line in enumerate(lines):
                    line_tokens = self._count_tokens(line + '\n')
                    if current_tokens + line_tokens > self.max_tokens and current_chunk:
                        yield CodeChunk(
                            id=self._generate_chunk_id(file_path),
                            content='\n'.join(current_chunk),
                            file_path=file_path,
                            language=language,
                            start_line=chunk_start,
                            end_line=chunk_start + len(current_chunk) - 1,
                            chunk_type="module_level_part",
                            imports=imports.copy(),
                        )
                        current_chunk = []
                        current_tokens = 0
                        chunk_start = start_line + i
                    
                    current_chunk.append(line)
                    current_tokens += line_tokens
                
                if current_chunk:
                    yield CodeChunk(
                        id=self._generate_chunk_id(file_path),
                        content='\n'.join(current_chunk),
                        file_path=file_path,
                        language=language,
                        start_line=chunk_start,
                        end_line=end_line,
                        chunk_type="module_level_part",
                        imports=imports.copy(),
                    )
    
    def _fallback_chunk(
        self,
        source: str,
        file_path: str,
        language: str
    ) -> Generator[CodeChunk, None, None]:
        """回退到简单的按行分块"""
        lines = source.split('\n')
        current_chunk = []
        current_tokens = 0
        chunk_start = 1
        
        for i, line in enumerate(lines, 1):
            line_tokens = self._count_tokens(line + '\n')
            
            if current_tokens + line_tokens > self.max_tokens and current_chunk:
                yield CodeChunk(
                    id=self._generate_chunk_id(file_path),
                    content='\n'.join(current_chunk),
                    file_path=file_path,
                    language=language,
                    start_line=chunk_start,
                    end_line=i - 1,
                    chunk_type="fallback",
                )
                
                # 保留overlap
                overlap_lines = current_chunk[-3:] if len(current_chunk) > 3 else current_chunk
                current_chunk = overlap_lines
                current_tokens = self._count_tokens('\n'.join(overlap_lines))
                chunk_start = i - len(overlap_lines)
            
            current_chunk.append(line)
            current_tokens += line_tokens
        
        if current_chunk:
            yield CodeChunk(
                id=self._generate_chunk_id(file_path),
                content='\n'.join(current_chunk),
                file_path=file_path,
                language=language,
                start_line=chunk_start,
                end_line=len(lines),
                chunk_type="fallback",
            )
    
    def chunk_directory(
        self,
        root_path: str,
        extensions: List[str] = None,
        exclude_patterns: List[str] = None
    ) -> Generator[CodeChunk, None, None]:
        """对整个目录进行分块"""
        import fnmatch
        
        if extensions is None:
            extensions = list(self.LANG_MAP.keys())
        if exclude_patterns is None:
            exclude_patterns = []
        
        root = Path(root_path)
        
        for file_path in root.rglob('*'):
            if not file_path.is_file():
                continue
            
            # 检查扩展名
            if file_path.suffix.lower() not in extensions:
                continue
            
            # 检查排除模式
            rel_path = str(file_path.relative_to(root))
            if any(fnmatch.fnmatch(rel_path, pattern) for pattern in exclude_patterns):
                continue
            
            yield from self.chunk_file(str(file_path))


# 使用示例
if __name__ == "__main__":
    chunker = CodeChunker(
        max_tokens=1024,
        overlap_tokens=128,
        include_context=True
    )
    
    # 分块单个文件
    # for chunk in chunker.chunk_file("example.cpp"):
    #     print(f"[{chunk.chunk_type}] {chunk.symbol_name}: {chunk.start_line}-{chunk.end_line}")
    #     print(chunk.full_content[:200])
    #     print("-" * 50)
    
    # 分块整个目录
    # for chunk in chunker.chunk_directory(
    #     "/path/to/project",
    #     extensions=[".cpp", ".h"],
    #     exclude_patterns=["**/build/**", "**/test/**"]
    # ):
    #     print(f"Processed: {chunk.file_path} - {chunk.chunk_type}")
