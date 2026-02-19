"""Project management for DevCrew."""

from datetime import datetime
from enum import Enum
from typing import Any

from devcrew.manager.task_tracker import Task, TaskStatus, TaskTracker, TaskType
from devcrew.persistence.file_store import FileStore


class ProjectStatus(str, Enum):
    """Project status enumeration."""

    CREATED = "created"
    PLANNING = "planning"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class Project:
    """Represents a development project managed by DevCrew."""

    def __init__(
        self,
        project_id: str,
        name: str,
        description: str = "",
        requirements: str = "",
        store: FileStore | None = None,
    ):
        self.project_id = project_id
        self.name = name
        self.description = description
        self.requirements = requirements
        self.status = ProjectStatus.CREATED
        self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()
        self.completed_at: str | None = None
        self.current_phase: str = "planning"

        self.store = store or FileStore()
        self.task_tracker = TaskTracker(project_id)
        self._load_state()

    def _load_state(self) -> None:
        """Load project state from storage."""
        state = self.store.load_state(self.project_id)
        if state:
            self.name = state.get("name", self.name)
            self.description = state.get("description", self.description)
            self.requirements = state.get("requirements", self.requirements)
            self.status = ProjectStatus(state.get("status", self.status.value))
            self.created_at = state.get("created_at", self.created_at)
            self.updated_at = state.get("updated_at", self.updated_at)
            self.completed_at = state.get("completed_at")
            self.current_phase = state.get("current_phase", self.current_phase)
            if "task_tracker" in state:
                self.task_tracker = TaskTracker.from_dict(self.project_id, state["task_tracker"])

    def save_state(self) -> None:
        """Save project state to storage."""
        self.updated_at = datetime.now().isoformat()
        state = {
            "project_id": self.project_id,
            "name": self.name,
            "description": self.description,
            "requirements": self.requirements,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "current_phase": self.current_phase,
            "task_tracker": self.task_tracker.to_dict(),
        }
        self.store.save_state(self.project_id, state)
        self.store.append_log(
            self.project_id,
            {"action": "state_saved", "status": self.status.value, "phase": self.current_phase},
        )

    def create_initial_tasks(self) -> None:
        """Create initial tasks for a new project."""
        tasks = [
            Task(
                id="req_analysis",
                title="需求分析",
                description="分析用户需求，转化为功能列表",
                task_type=TaskType.REQUIREMENT,
                assignee="需求分析师",
            ),
            Task(
                id="architecture",
                title="架构设计",
                description="设计技术方案，选择技术栈",
                task_type=TaskType.ARCHITECTURE,
                assignee="架构设计师",
                dependencies=["req_analysis"],
            ),
            Task(
                id="implementation",
                title="代码实现",
                description="根据设计文档实现代码",
                task_type=TaskType.IMPLEMENTATION,
                assignee="开发者",
                dependencies=["architecture"],
            ),
            Task(
                id="testing",
                title="测试验证",
                description="编写测试用例，验证功能",
                task_type=TaskType.TESTING,
                assignee="测试工程师",
                dependencies=["implementation"],
            ),
            Task(
                id="documentation",
                title="文档整理",
                description="整理项目文档",
                task_type=TaskType.DOCUMENTATION,
                assignee="文档工程师",
                dependencies=["testing"],
            ),
        ]
        for task in tasks:
            self.task_tracker.add_task(task)
        self.status = ProjectStatus.PLANNING
        self.save_state()

    def get_overall_progress(self) -> dict[str, Any]:
        """Get overall project progress."""
        task_progress = self.task_tracker.get_progress()
        percentage: float = (
            round(task_progress["completed"] / task_progress["total"] * 100, 1)
            if task_progress["total"] > 0
            else 0.0
        )
        task_progress["percentage"] = percentage
        return {
            "project_id": self.project_id,
            "name": self.name,
            "status": self.status.value,
            "current_phase": self.current_phase,
            "tasks": task_progress,
        }

    def get_task_tree(self) -> list[dict[str, Any]]:
        """Get task tree with hierarchy."""
        tasks = []
        for task in self.task_tracker.tasks.values():
            tasks.append(task.to_dict())
        return tasks

    def to_dict(self) -> dict[str, Any]:
        """Convert project to dictionary."""
        return {
            "project_id": self.project_id,
            "name": self.name,
            "description": self.description,
            "requirements": self.requirements,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "current_phase": self.current_phase,
            "progress": self.get_overall_progress(),
        }

    @classmethod
    def load(cls, project_id: str, store: FileStore | None = None) -> "Project | None":
        """Load an existing project."""
        store = store or FileStore()
        if not store.project_exists(project_id):
            return None
        project = cls(project_id, "", store=store)
        return project

    @classmethod
    def create(
        cls,
        project_id: str,
        name: str,
        description: str = "",
        requirements: str = "",
        store: FileStore | None = None,
    ) -> "Project":
        """Create a new project."""
        project = cls(project_id, name, description, requirements, store)
        project.store.ensure_project_dirs(project_id)
        project.create_initial_tasks()
        project.save_state()
        project.store.save_document(project_id, "SPEC.md", f"# {name}\n\n{description}\n\n## 需求\n{requirements}")
        return project

    def pause(self) -> None:
        """Pause the project."""
        self.status = ProjectStatus.PAUSED
        self.save_state()

    def resume(self) -> None:
        """Resume the project."""
        self.status = ProjectStatus.IN_PROGRESS
        self.save_state()

    def complete(self) -> None:
        """Complete the project."""
        self.status = ProjectStatus.COMPLETED
        self.completed_at = datetime.now().isoformat()
        self.save_state()
