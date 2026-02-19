---
name: devcrew
description: Multi-Agent Development Team using CrewAI. Use when you need to manage a team of AI agents that can analyze requirements, design architecture, implement code, and test software. Supports pausing/resuming and provides Web/TUI interfaces for monitoring.
---

# DevCrew - 多智能体开发团队

基于CrewAI的多智能体开发团队框架，支持需求分析、架构设计、代码实现、测试验证等完整开发流程。

## 核心功能

- **多Agent协作**: 需求分析师、架构设计师、开发者、测试工程师、文档工程师
- **任务追踪**: 实时追踪每个子任务进度和总体进度
- **持久化存储**: 支持暂停后继续执行，状态自动保存到文件系统
- **文档自动生成**: 自动生成SPEC.md、PROGRESS.md、README.md
- **双界面支持**: Web界面 + 终端界面(TUI)

## 使用方式

### 1. 作为Skill调用 (推荐)

```python
from crewai import Agent

devcrew_agent = Agent(
    role="DevCrew Manager",
    goal="管理和协调多智能体开发团队",
    backstory="你是一个开发团队管理器，擅长协调多个AI Agent完成软件开发任务。",
    tools=[DevCrewTool()]  # 使用DevCrewTool工具
)
```

### 2. Python代码调用

```python
from devcrew import Project, DevCrew
from devcrew.persistence import FileStore

# 创建项目
store = FileStore("./projects")
project = Project.create(
    project_id="my_project",
    name="我的项目",
    description="这是一个测试项目",
    requirements="实现一个简单的计算器",
    store=store
)

# 运行开发团队
crew = DevCrew(project, model="gpt-4o")
results = crew.run()

# 查看进度
print(project.get_overall_progress())
```

### 3. 使用Web界面

```bash
# 安装依赖
pip install -e .

# 启动Web界面
python -m devcrew.ui.web

# 访问 http://localhost:5000
```

### 4. 使用终端界面

```bash
# 启动TUI
python -m devcrew.ui.tui
```

## Agent角色

| Agent | 职责 |
|-------|------|
| **需求分析师** | 深入理解用户需求，转化为功能列表 |
| **架构设计师** | 设计技术方案，选择技术栈 |
| **开发者** | 根据设计文档实现代码 |
| **测试工程师** | 编写测试用例，验证功能 |
| **文档工程师** | 整理项目文档 |

## 项目结构

```
projects/
├── {project_id}/
│   ├── state.json          # 项目状态
│   ├── docs/
│   │   ├── SPEC.md         # 需求规格
│   │   ├── PROGRESS.md     # 进度报告
│   │   └── README.md       # 项目说明
│   ├── logs/
│   │   └── activity.log    # 活动日志
│   ├── src/                # 源代码
│   └── tests/              # 测试代码
```

## 暂停/继续功能

项目状态自动保存到文件系统，暂停后可通过以下方式继续：

1. **Web界面**: 点击"继续"按钮
2. **API调用**: `POST /api/projects/{id}/resume`
3. **代码调用**: `project.resume()` + `crew.run()`

## 配置

### LLM配置

```python
from crewai import LLM
from devcrew import DevCrew

# 使用不同的LLM
crew = DevCrew(project, model="claude-sonnet-4-20250514")
crew = DevCrew(project, model="ollama/llama3.1", base_url="http://localhost:11434")
```

### 存储路径配置

```python
from devcrew.persistence import FileStore

store = FileStore("/custom/path/to/projects")
project = Project.create(..., store=store)
```

## 依赖

- crewai>=0.70.0
- crewai-tools>=0.10.0
- flask>=3.0.0
- textual>=0.80.0
- pydantic>=2.0.0

## 注意事项

1. 首次使用需配置LLM API Key (OPENAI_API_KEY, ANTHROPIC_API_KEY等)
2. 建议在执行长时间任务前确保网络连接稳定
3. 项目文件默认保存在 `./projects` 目录
