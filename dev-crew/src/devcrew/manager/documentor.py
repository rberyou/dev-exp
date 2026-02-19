"""Documentor module for DevCrew."""

from datetime import datetime
from typing import Any

from devcrew.manager.project import Project
from devcrew.manager.task_tracker import TaskStatus


class Documentor:
    """Generates and manages project documentation."""

    def __init__(self, project: Project):
        """Initialize the documentor.

        Args:
            project: Project instance to document
        """
        self.project = project

    def generate_progress_report(self) -> str:
        """Generate a progress report in Markdown format."""
        progress = self.project.get_overall_progress()
        tasks = self.project.get_task_tree()

        report = f"""# 项目进度报告

## 项目信息
- **项目名称**: {self.project.name}
- **项目ID**: {self.project.project_id}
- **状态**: {progress['status']}
- **当前阶段**: {progress['current_phase']}
- **创建时间**: {self.project.created_at}
- **更新时间**: {self.project.updated_at}

## 进度概览
- **总体进度**: {progress['tasks']['percentage']}%
- **总任务数**: {progress['tasks']['total']}
- **已完成**: {progress['tasks']['completed']}
- **进行中**: {progress['tasks']['in_progress']}
- **待处理**: {progress['tasks']['pending']}
- **失败**: {progress['tasks']['failed']}

## 任务列表

| 任务ID | 任务名称 | 类型 | 状态 | 负责人 |
|--------|----------|------|------|--------|
"""
        for task in tasks:
            status_emoji = {
                TaskStatus.COMPLETED.value: "✅",
                TaskStatus.IN_PROGRESS.value: "🔄",
                TaskStatus.PENDING.value: "⏳",
                TaskStatus.FAILED.value: "❌",
                TaskStatus.BLOCKED.value: "🚫",
            }.get(task["status"], "❓")

            report += f"| {task['id']} | {task['title']} | {task['task_type']} | {status_emoji} {task['status']} | {task.get('assignee', '-')} |\n"

        report += f"""

## 时间线
- **开始时间**: {self.project.created_at}
"""
        if self.project.completed_at:
            report += f"- **完成时间**: {self.project.completed_at}\n"

        report += f"- **最后更新**: {self.project.updated_at}\n"

        return report

    def generate_spec(self) -> str:
        """Generate or update the SPEC.md document."""
        req_task = self.project.task_tracker.get_task("req_analysis")
        arch_task = self.project.task_tracker.get_task("architecture")
        impl_task = self.project.task_tracker.get_task("implementation")
        test_task = self.project.task_tracker.get_task("testing")
        doc_task = self.project.task_tracker.get_task("documentation")

        spec = f"""# {self.project.name} - 规格说明书

## 项目概述
{self.project.description}

## 需求

### 功能需求
{self.project.requirements}

## 开发状态

### 当前阶段
{self.project.current_phase}

### 进度
- 需求分析: {"✅" if req_task and req_task.status == TaskStatus.COMPLETED else "⏳"}
- 架构设计: {"✅" if arch_task and arch_task.status == TaskStatus.COMPLETED else "⏳"}
- 代码实现: {"✅" if impl_task and impl_task.status == TaskStatus.COMPLETED else "⏳"}
- 测试验证: {"✅" if test_task and test_task.status == TaskStatus.COMPLETED else "⏳"}
- 文档整理: {"✅" if doc_task and doc_task.status == TaskStatus.COMPLETED else "⏳"}

---
*最后更新: {datetime.now().isoformat()}*
"""
        return spec

    def generate_readme(self) -> str:
        """Generate a README.md for the project."""
        progress = self.project.get_overall_progress()

        req_task = self.project.task_tracker.get_task("req_analysis")
        arch_task = self.project.task_tracker.get_task("architecture")
        impl_task = self.project.task_tracker.get_task("implementation")
        test_task = self.project.task_tracker.get_task("testing")
        doc_task = self.project.task_tracker.get_task("documentation")

        readme = f"""# {self.project.name}

{self.project.description}

## 项目状态

| 指标 | 值 |
|------|-----|
| 进度 | {progress['tasks']['percentage']}% |
| 状态 | {progress['status']} |
| 阶段 | {progress['current_phase']} |

## 任务进度

- ✅ 需求分析: {"完成" if req_task and req_task.status == TaskStatus.COMPLETED else "进行中/待处理"}
- ✅ 架构设计: {"完成" if arch_task and arch_task.status == TaskStatus.COMPLETED else "进行中/待处理"}
- ✅ 代码实现: {"完成" if impl_task and impl_task.status == TaskStatus.COMPLETED else "进行中/待处理"}
- ✅ 测试验证: {"完成" if test_task and test_task.status == TaskStatus.COMPLETED else "进行中/待处理"}
- ✅ 文档整理: {"完成" if doc_task and doc_task.status == TaskStatus.COMPLETED else "进行中/待处理"}

## 快速开始

```bash
# 启动开发团队
python -m devcrew.ui.cli start {self.project.project_id}
```

## 查看进度

```bash
# Web界面
python -m devcrew.ui.web

# 终端界面
python -m devcrew.ui.tui
```

---
*由 DevCrew 自动生成*
"""
        return readme

    def save_all_documents(self) -> dict[str, str]:
        """Save all documents to storage."""
        docs = {
            "PROGRESS.md": self.generate_progress_report(),
            "SPEC.md": self.generate_spec(),
            "README.md": self.generate_readme(),
        }

        for doc_name, content in docs.items():
            self.project.store.save_document(self.project.project_id, doc_name, content)

        return docs
