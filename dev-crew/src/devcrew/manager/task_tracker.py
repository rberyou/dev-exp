"""Task tracking system for DevCrew."""

from datetime import datetime
from enum import Enum
from typing import Any


class TaskStatus(str, Enum):
    """Task status enumeration."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class TaskType(str, Enum):
    """Task type enumeration."""

    REQUIREMENT = "requirement"
    ARCHITECTURE = "architecture"
    IMPLEMENTATION = "implementation"
    TESTING = "testing"
    REVIEW = "review"
    DOCUMENTATION = "documentation"


class Task:
    """Represents a task in the development process."""

    def __init__(
        self,
        id: str,
        title: str,
        description: str,
        task_type: TaskType,
        status: TaskStatus = TaskStatus.PENDING,
        assignee: str | None = None,
        dependencies: list[str] | None = None,
        output: str | None = None,
        error: str | None = None,
    ):
        self.id = id
        self.title = title
        self.description = description
        self.task_type = task_type
        self.status = status
        self.assignee = assignee
        self.dependencies = dependencies or []
        self.output = output
        self.error = error
        self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()
        self.completed_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert task to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "task_type": self.task_type.value,
            "status": self.status.value,
            "assignee": self.assignee,
            "dependencies": self.dependencies,
            "output": self.output,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Task":
        """Create task from dictionary."""
        task = cls(
            id=data["id"],
            title=data["title"],
            description=data["description"],
            task_type=TaskType(data["task_type"]),
            status=TaskStatus(data["status"]),
            assignee=data.get("assignee"),
            dependencies=data.get("dependencies", []),
            output=data.get("output"),
            error=data.get("error"),
        )
        task.created_at = data.get("created_at", task.created_at)
        task.updated_at = data.get("updated_at", task.updated_at)
        task.completed_at = data.get("completed_at")
        return task

    def update_status(self, status: TaskStatus, output: str | None = None, error: str | None = None) -> None:
        """Update task status."""
        self.status = status
        self.updated_at = datetime.now().isoformat()
        if output:
            self.output = output
        if error:
            self.error = error
        if status == TaskStatus.COMPLETED:
            self.completed_at = datetime.now().isoformat()


class TaskTracker:
    """Tracks tasks for a project."""

    def __init__(self, project_id: str):
        self.project_id = project_id
        self.tasks: dict[str, Task] = {}

    def add_task(self, task: Task) -> None:
        """Add a task to the tracker."""
        self.tasks[task.id] = task

    def get_task(self, task_id: str) -> Task | None:
        """Get a task by ID."""
        return self.tasks.get(task_id)

    def get_tasks_by_status(self, status: TaskStatus) -> list[Task]:
        """Get all tasks with a specific status."""
        return [t for t in self.tasks.values() if t.status == status]

    def get_tasks_by_type(self, task_type: TaskType) -> list[Task]:
        """Get all tasks of a specific type."""
        return [t for t in self.tasks.values() if t.task_type == task_type]

    def get_ready_tasks(self) -> list[Task]:
        """Get tasks that are ready to execute (pending with no blocked dependencies)."""
        ready = []
        for task in self.tasks.values():
            if task.status != TaskStatus.PENDING:
                continue
            deps_completed = all(
                self.tasks.get(dep_id) and self.tasks[dep_id].status == TaskStatus.COMPLETED
                for dep_id in task.dependencies
            )
            if deps_completed:
                ready.append(task)
        return ready

    def get_progress(self) -> dict[str, int | float]:
        """Get progress statistics."""
        total = len(self.tasks)
        if total == 0:
            return {"total": 0, "completed": 0, "in_progress": 0, "pending": 0, "failed": 0}
        return {
            "total": total,
            "completed": len(self.get_tasks_by_status(TaskStatus.COMPLETED)),
            "in_progress": len(self.get_tasks_by_status(TaskStatus.IN_PROGRESS)),
            "pending": len(self.get_tasks_by_status(TaskStatus.PENDING)),
            "failed": len(self.get_tasks_by_status(TaskStatus.FAILED)),
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert tracker to dictionary."""
        return {
            "project_id": self.project_id,
            "tasks": {task_id: task.to_dict() for task_id, task in self.tasks.items()},
        }

    @classmethod
    def from_dict(cls, project_id: str, data: dict[str, Any]) -> "TaskTracker":
        """Create tracker from dictionary."""
        tracker = cls(project_id)
        for task_id, task_data in data.get("tasks", {}).items():
            tracker.tasks[task_id] = Task.from_dict(task_data)
        return tracker
