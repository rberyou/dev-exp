# Code RAG System 安装指南

## 解决 torch 依赖冲突问题

由于 vLLM 0.2.7 版本与某些 torch 版本存在兼容性问题，我们提供了以下解决方案：

### 方案一：使用 Ollama（推荐）

1. **安装基础依赖**（不含 vLLM）：
   ```bash
   pip install -r requirements.txt
   ```

2. **安装 Ollama**（如果尚未安装）：
   - 访问 https://ollama.com/ 下载并安装 Ollama
   - 或使用命令行安装：
     ```bash
     # Linux/macOS
     curl -fsSL https://ollama.ai/install.sh | sh
     ```

3. **启动 Ollama 服务**：
   ```bash
   ollama serve
   ```

4. **拉取所需的模型**：
   ```bash
   ollama pull qwen2.5-coder:14b-instruct-q8_0
   # 或者其他你喜欢的模型，如 llama3:8b, mistral:7b 等
   ```

5. **运行系统**：
   ```bash
   python run_with_ollama.py
   ```

### 方案二：升级 vLLM 版本

如果你想继续使用 vLLM，可以尝试升级到较新的版本：

1. 修改 requirements.txt 文件，将：
   ```
   vllm==0.2.7
   ```
   替换为：
   ```
   vllm>=0.5.0
   ```

2. 然后安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

### 方案三：使用 Conda 环境

如果仍然遇到 torch 版本问题，建议使用 conda 来更好地管理依赖：

```bash
# 创建新的conda环境
conda create -n code-rag python=3.10
conda activate code-rag

# 安装 torch（选择适合你的硬件的版本）
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 或 CUDA 12.1
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 然后安装其他依赖
pip install -r requirements.txt
```

## 使用说明

- 默认配置文件是 `config.yaml`
- 使用 Ollama 的配置文件是 `config-ollama.yaml`
- 快速启动脚本：`python quickstart.py`
- 使用 Ollama 的启动脚本：`python run_with_ollama.py`
- 模型预下载脚本：`python download_models.py`

## 注意事项

1. 如果使用 Ollama，请确保 Ollama 服务在运行状态
2. 检查模型是否已正确下载
3. 确保防火墙或网络设置允许本地 API 调用
4. 如遇到其他依赖冲突，建议使用虚拟环境隔离安装

## 模型缓存管理

模型默认下载到 `~/.cache/huggingface/hub/` 目录。如果需要更改下载位置：

1. **临时设置**（仅本次运行有效）：
   ```bash
   HF_HOME=/path/to/your/cache python download_models.py
   ```

2. **永久设置**（在 `download_models.py` 中修改）：
   ```python
   # 取消注释并修改以下行
   MODEL_CACHE_DIR = "/mnt/your_large_disk/hf_models"
   os.environ['HF_HOME'] = MODEL_CACHE_DIR
   ```

3. **环境变量设置**（添加到 `.bashrc` 或 `.zshrc`）：
   ```bash
   export HF_HOME=/path/to/your/cache
   ```

**重要**：模型一旦下载到指定位置，其他代码会自动从缓存加载，不会重复下载。

## 环境变量配置

我们已经将 `HF_HOME` 环境变量添加到 `~/.bashrc` 中，指向 `/mnt/d/Workspace/ReservedForLinux/hf_home_cache` 目录。这意味着：

- 新的 Hugging Face 模型将下载到 `/mnt/d/Workspace/ReservedForLinux/hf_home_cache` 目录
- 已迁移的模型也将从此位置加载
- 此设置在每次打开新终端时自动生效

要验证设置是否生效，可以运行：
```bash
source ~/.bashrc && echo $HF_HOME
```