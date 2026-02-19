# DevCrew 多智能体开发团队

## 概述

DevCrew 是一个基于 CrewAI 的多智能体软件开发框架，通过多个专业 AI Agent 协作完成从需求分析到测试部署的完整开发流程。

### 核心特性

- **多Agent协作**: 需求分析师 → 架构设计师 → 开发者 → 测试工程师 → 文档工程师
- **持久化支持**: 随时暂停/继续开发进度，状态自动保存
- **文档自动生成**: 自动生成 SPEC.md、PROGRESS.md、README.md
- **双界面**: Web界面 + 终端界面(TUI)
- **Skill封装**: 可被其他 Agent 调用

---

## 架构设计

### 模块结构

```
src/devcrew/
├── __init__.py           # 模块入口，导出核心类
├── cli.py                # 命令行入口
├── crew/                 # CrewAI 核心
│   ├── agents.py        # Agent 定义 (6种角色)
│   ├── tasks.py        # Task 定义 (5种任务)
│   └── crew.py         # Crew 编排逻辑
├── manager/             # 项目管理
│   ├── project.py      # Project 类 (项目管理核心)
│   ├── task_tracker.py # Task/TaskTracker (任务追踪)
│   └── documentor.py   # 文档生成器
├── persistence/         # 持久化层
│   └── file_store.py   # FileStore (文件系统存储)
└── ui/                  # 用户界面
    ├── web.py          # Flask Web 界面
    └── tui.py          # Textual 终端界面
```

### 数据流

```
用户请求
    ↓
CLI/Web/TUI
    ↓
Project (加载或创建)
    ↓
DevCrew (创建Crew)
    ↓
CrewAI Agents (执行任务)
    ↓
TaskTracker (更新状态) → FileStore (持久化)
    ↓
Documentor (生成文档)
```

---

## 核心类说明

### 1. Project (`manager/project.py`)

项目管理的核心类。

**属性:**
- `project_id`: 项目唯一标识
- `name`: 项目名称
- `description`: 项目描述
- `requirements`: 需求描述
- `status`: 项目状态 (created/planning/in_progress/paused/completed/failed)
- `current_phase`: 当前阶段
- `task_tracker`: TaskTracker 实例

**方法:**
- `create()`: 静态方法，创建新项目
- `load()`: 静态方法，加载已有项目
- `save_state()`: 保存项目状态到文件
- `pause()`: 暂停项目
- `resume()`: 恢复项目
- `complete()`: 完成项目
- `get_overall_progress()`: 获取总体进度
- `get_task_tree()`: 获取任务树

### 2. TaskTracker (`manager/task_tracker.py`)

任务追踪器。

**方法:**
- `add_task(task)`: 添加任务
- `get_task(task_id)`: 获取任务
- `get_tasks_by_status(status)`: 按状态筛选
- `get_ready_tasks()`: 获取可执行的任务
- `get_progress()`: 获取进度统计

### 3. Task (`manager/task_tracker.py`)

单个任务。

**属性:**
- `id`: 任务ID
- `title`: 任务标题
- `description`: 任务描述
- `task_type`: 任务类型 (requirement/architecture/implementation/testing/documentation)
- `status`: 任务状态 (pending/in_progress/completed/failed/blocked)
- `assignee`: 负责人
- `dependencies`: 依赖的任务ID列表
- `output`: 输出内容
- `error`: 错误信息

### 4. DevCrew (`crew/crew.py`)

Crew 编排类。

**方法:**
- `__init__(project, llm, model)`: 初始化
- `run(start_phase)`: 运行完整流程
- `run_phase(phase)`: 运行特定阶段
- `get_status()`: 获取状态

### 5. FileStore (`persistence/file_store.py`)

文件系统存储。

**方法:**
- `save_state(project_id, state)`: 保存状态
- `load_state(project_id)`: 加载状态
- `save_document(project_id, doc_name, content)`: 保存文档
- `load_document(project_id, doc_name)`: 加载文档
- `append_log(project_id, log_entry)`: 追加日志
- `list_projects()`: 列出所有项目

---

## Agent 定义 (`crew/agents.py`)

| Agent | 角色 | 目标 |
|-------|------|------|
| `requirements_analyst()` | 需求分析师 | 深入理解用户需求，转化为功能列表 |
| `architect()` | 架构设计师 | 设计技术方案，选择技术栈 |
| `developer()` | 开发者 | 实现高质量代码 |
| `tester()` | 测试工程师 | 编写测试用例，验证功能 |
| `documenter()` | 文档工程师 | 创建完整项目文档 |
| `manager()` | 项目经理 | 协调各个阶段，管理进度 |

每个 Agent 使用 `llm` 参数指定 LLM，可用的模型包括:
- OpenAI: `gpt-4o`, `gpt-4-turbo`
- Anthropic: `claude-sonnet-4-20250514`
- 本地: `ollama/llama3.1`

---

## Task 定义 (`crew/tasks.py`)

| Task | Agent | 输入 | 输出 |
|------|-------|------|------|
| `requirements_analysis()` | requirements_analyst | requirements, context | 需求分析报告 |
| `architecture_design()` | architect | requirements, context | 架构设计文档 |
| `implementation()` | developer | architecture, context | 实现代码 |
| `testing()` | tester | implementation, requirements | 测试报告 |
| `documentation()` | documenter | context | 项目文档 |

---

## 持久化结构

```
projects/
└── {project_id}/
    ├── state.json          # 项目状态 (JSON)
    ├── docs/
    │   ├── SPEC.md         # 需求规格
    │   ├── PROGRESS.md     # 进度报告
    │   └── README.md       # 项目说明
    ├── logs/
    │   └── activity.log    # 活动日志 (JSONL)
    ├── src/                # 源代码
    └── tests/              # 测试代码
```

### state.json 结构

```json
{
  "project_id": "my_project",
  "name": "我的项目",
  "description": "...",
  "requirements": "...",
  "status": "in_progress",
  "current_phase": "implementation",
  "task_tracker": {
    "project_id": "my_project",
    "tasks": {
      "req_analysis": {
        "id": "req_analysis",
        "title": "需求分析",
        "status": "completed",
        "output": "..."
      }
    }
  }
}
```

---

## CLI 命令

### 创建项目
```bash
python -m devcrew.cli create \
  --id my_app \
  --name "我的应用" \
  --description "一个测试项目" \
  --requirements "实现一个待办事项应用"
```

### 查看项目列表
```bash
python -m devcrew.cli list
```

### 查看项目状态
```bash
python -m devcrew.cli status my_app
```

### 启动项目开发
```bash
python -m devcrew.cli start my_app --model gpt-4o
```

### 运行特定阶段
```bash
python -m devcrew.cli phase my_app requirements
python -m devcrew.cli phase my_app architecture
python -m devcrew.cli phase my_app implementation
python -m devcrew.cli phase my_app testing
python -m devcrew.cli phase my_app documentation
```

### 启动界面
```bash
python -m devcrew.cli web --port 5000
python -m devcrew.cli tui
```

---

## Web API

### 端点列表

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 仪表盘 |
| GET | `/project/<id>` | 项目详情 |
| GET | `/api/projects` | 获取项目列表 |
| POST | `/api/projects` | 创建项目 |
| GET | `/api/projects/<id>` | 获取项目详情 |
| GET | `/api/projects/<id>/progress` | 获取进度 |
| GET | `/api/projects/<id>/tasks` | 获取任务列表 |
| POST | `/api/projects/<id>/start` | 启动项目 |
| POST | `/api/projects/<id>/phase/<phase>` | 运行阶段 |
| POST | `/api/projects/<id>/pause` | 暂停项目 |
| POST | `/api/projects/<id>/resume` | 恢复项目 |

### API 使用示例

```bash
# 创建项目
curl -X POST http://localhost:5000/api/projects \
  -H "Content-Type: application/json" \
  -d '{"name": "测试", "requirements": "实现计算器"}'

# 获取项目进度
curl http://localhost:5000/api/projects/my_app/progress
```

---

## 扩展开发

### 添加新的 Agent

在 `crew/agents.py` 中添加方法:

```python
@staticmethod
def new_agent(tools: list[BaseTool] | None = None, **kwargs) -> Agent:
    return Agent(
        role="新角色",
        goal="目标描述",
        backstory="背景故事",
        verbose=True,
        tools=tools or [],
        **kwargs,
    )
```

### 添加新的 Task

在 `crew/tasks.py` 中添加方法:

```python
@staticmethod
def new_task(agent: Agent, context: str = "", **kwargs) -> Task:
    return Task(
        description="任务描述",
        expected_output="期望输出",
        agent=agent,
        **kwargs,
    )
```

### 添加新的阶段

在 `crew/crew.py` 的 `_run_phase()` 中添加:

```python
elif phase == "new_phase":
    task = self.tasks.new_task(agent=agents["new_agent"], ...)
    # 执行逻辑...
```

### 使用自定义存储

```python
from devcrew.persistence.file_store import FileStore
from devcrew import Project

store = FileStore("/custom/path")
project = Project.create(..., store=store)
```

---

## 依赖

```
crewai>=0.70.0
crewai-tools>=0.10.0
flask>=3.0.0
textual>=0.80.0
pydantic>=2.0.0
python-dotenv>=1.0.0
jinja2>=3.1.0
markdown>=3.5.0
aiohttp>=3.9.0
```

---

## 常见问题

### 1. 安装后无法导入

```bash
pip install -e .
# 或
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
```

### 2. LLM API 错误

确保环境变量已设置:
- OpenAI: `OPENAI_API_KEY`
- Anthropic: `ANTHROPIC_API_KEY`

### 3. 任务卡住

检查 `crew/crew.py` 中的 `max_iter` 参数，默认 15 次迭代。

### 4. 状态不同步

手动调用 `project.save_state()` 保存状态。

---

## 文件清单

| 文件 | 行数 | 说明 |
|------|------|------|
| `__init__.py` | ~15 | 模块入口 |
| `cli.py` | ~150 | 命令行入口 |
| `crew/agents.py` | ~130 | Agent 定义 |
| `crew/tasks.py` | ~160 | Task 定义 |
| `crew/crew.py` | ~180 | Crew 编排 |
| `manager/project.py` | ~200 | 项目管理 |
| `manager/task_tracker.py` | ~170 | 任务追踪 |
| `manager/documentor.py` | ~160 | 文档生成 |
| `persistence/file_store.py` | ~100 | 文件存储 |
| `ui/web.py` | ~155 | Web 界面 |
| `ui/tui.py` | ~200 | 终端界面 |
