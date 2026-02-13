from crewai.tools import tool
from typing import Optional
import os

@tool("Write Code File")
def write_code_file(file_path: str, content: str, description: Optional[str] = None) -> str:
    """
    Write code content to a file in the output directory.
    
    Args:
        file_path: Relative path for the file (e.g., 'app/page.tsx')
        content: The code content to write
        description: Optional description of what this file contains
    
    Returns:
        Success message with the full file path
    """
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
    full_path = os.path.join(output_dir, file_path)
    
    dir_path = os.path.dirname(full_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    
    with open(full_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return f"Successfully wrote file: {full_path}"

@tool("Create Directory Structure")
def create_directory(dir_path: str) -> str:
    """
    Create a directory structure in the output folder.
    
    Args:
        dir_path: Relative path for the directory (e.g., 'app/components')
    
    Returns:
        Success message with the full directory path
    """
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
    full_path = os.path.join(output_dir, dir_path)
    
    os.makedirs(full_path, exist_ok=True)
    
    return f"Successfully created directory: {full_path}"

@tool("Read Code File")
def read_code_file(file_path: str) -> str:
    """
    Read content from a file in the output directory.
    
    Args:
        file_path: Relative path for the file
    
    Returns:
        The content of the file or error message
    """
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
    full_path = os.path.join(output_dir, file_path)
    
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return f"File not found: {full_path}"

@tool("List Output Files")
def list_output_files(dir_path: str = "") -> str:
    """
    List all files in the output directory.
    
    Args:
        dir_path: Relative path to list (empty for root)
    
    Returns:
        List of files and directories
    """
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
    full_path = os.path.join(output_dir, dir_path) if dir_path else output_dir
    
    if not os.path.exists(full_path):
        return f"Directory not found: {full_path}"
    
    items = []
    for item in os.listdir(full_path):
        item_path = os.path.join(full_path, item)
        if os.path.isdir(item_path):
            items.append(f"[DIR]  {item}/")
        else:
            items.append(f"[FILE] {item}")
    
    return "\n".join(items) if items else "Empty directory"
