from crewai.tools import tool
import os
import json
from typing import Optional, Dict, Any

@tool("Initialize Next.js Project")
def init_nextjs_project(project_name: str = "website") -> str:
    """
    Initialize a Next.js project structure with TypeScript and Tailwind CSS.
    
    Args:
        project_name: Name of the project
    
    Returns:
        Success message with created structure
    """
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
    
    directories = [
        project_name,
        f"{project_name}/app",
        f"{project_name}/app/api",
        f"{project_name}/app/components",
        f"{project_name}/app/lib",
        f"{project_name}/app/types",
        f"{project_name}/public",
    ]
    
    for dir_path in directories:
        os.makedirs(os.path.join(output_dir, dir_path), exist_ok=True)
    
    package_json = {
        "name": project_name,
        "version": "0.1.0",
        "private": True,
        "scripts": {
            "dev": "next dev",
            "build": "next build",
            "start": "next start",
            "lint": "next lint",
            "test": "jest"
        },
        "dependencies": {
            "next": "^14.0.0",
            "react": "^18.2.0",
            "react-dom": "^18.2.0",
            "tailwindcss": "^3.4.0",
            "typescript": "^5.0.0"
        },
        "devDependencies": {
            "@types/node": "^20.0.0",
            "@types/react": "^18.2.0",
            "@types/react-dom": "^18.2.0",
            "autoprefixer": "^10.4.0",
            "postcss": "^8.4.0",
            "eslint": "^8.0.0",
            "eslint-config-next": "^14.0.0"
        }
    }
    
    package_path = os.path.join(output_dir, project_name, "package.json")
    with open(package_path, 'w', encoding='utf-8') as f:
        json.dump(package_json, f, indent=2)
    
    return f"Initialized Next.js project '{project_name}' with structure:\n" + "\n".join(directories)

@tool("Create Config File")
def create_config_file(config_type: str, project_name: str = "website") -> str:
    """
    Create configuration files for Next.js project.
    
    Args:
        config_type: Type of config (tailwind, next, tsconfig, postcss)
        project_name: Name of the project
    
    Returns:
        Success message
    """
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
    project_dir = os.path.join(output_dir, project_name)
    
    configs = {
        "tailwind": {
            "filename": "tailwind.config.js",
            "content": """/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './app/components/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
"""
        },
        "next": {
            "filename": "next.config.js",
            "content": """/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
}

module.exports = nextConfig
"""
        },
        "tsconfig": {
            "filename": "tsconfig.json",
            "content": """{
  "compilerOptions": {
    "target": "es5",
    "lib": ["dom", "dom.iterable", "esnext"],
    "allowJs": true,
    "skipLibCheck": true,
    "strict": true,
    "noEmit": true,
    "esModuleInterop": true,
    "module": "esnext",
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "jsx": "preserve",
    "incremental": true,
    "plugins": [{"name": "next"}],
    "paths": {
      "@/*": ["./*"]
    }
  },
  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx", ".next/types/**/*.ts"],
  "exclude": ["node_modules"]
}
"""
        },
        "postcss": {
            "filename": "postcss.config.js",
            "content": """module.exports = {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}
"""
        }
    }
    
    if config_type not in configs:
        return f"Unknown config type: {config_type}. Available: {list(configs.keys())}"
    
    config = configs[config_type]
    file_path = os.path.join(project_dir, config["filename"])
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(config["content"])
    
    return f"Created {config['filename']}"

@tool("Get Project Stats")
def get_project_stats(project_name: str = "website") -> str:
    """
    Get statistics about the generated project.
    
    Args:
        project_name: Name of the project
    
    Returns:
        Project statistics
    """
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
    project_dir = os.path.join(output_dir, project_name)
    
    if not os.path.exists(project_dir):
        return f"Project '{project_name}' not found"
    
    stats = {
        "total_files": 0,
        "total_dirs": 0,
        "file_types": {},
        "files_by_dir": {}
    }
    
    for root, dirs, files in os.walk(project_dir):
        stats["total_dirs"] += len(dirs)
        stats["total_files"] += len(files)
        
        rel_dir = os.path.relpath(root, project_dir)
        if files:
            stats["files_by_dir"][rel_dir] = files
        
        for file in files:
            ext = os.path.splitext(file)[1]
            stats["file_types"][ext] = stats["file_types"].get(ext, 0) + 1
    
    return json.dumps(stats, indent=2)
