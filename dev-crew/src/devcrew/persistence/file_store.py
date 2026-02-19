"""File-based persistence layer for DevCrew."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


class FileStore:
    """File-based storage for project state and documents."""

    def __init__(self, base_path: str | None = None):
        """Initialize the file store.

        Args:
            base_path: Base directory for storing projects. Defaults to ./projects
        """
        self.base_path = Path(base_path) if base_path else Path.cwd() / "projects"
        self.base_path.mkdir(parents=True, exist_ok=True)

    def get_project_path(self, project_id: str) -> Path:
        """Get the path for a project directory."""
        return self.base_path / project_id

    def ensure_project_dirs(self, project_id: str) -> dict[str, Path]:
        """Ensure project directories exist and return paths."""
        project_path = self.get_project_path(project_id)
        dirs = {
            "root": project_path,
            "docs": project_path / "docs",
            "logs": project_path / "logs",
            "src": project_path / "src",
            "tests": project_path / "tests",
        }
        for dir_path in dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        return dirs

    def save_state(self, project_id: str, state: dict[str, Any]) -> None:
        """Save project state to JSON file."""
        dirs = self.ensure_project_dirs(project_id)
        state_file = dirs["root"] / "state.json"
        state["updated_at"] = datetime.now().isoformat()
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    def load_state(self, project_id: str) -> dict[str, Any] | None:
        """Load project state from JSON file."""
        state_file = self.get_project_path(project_id) / "state.json"
        if not state_file.exists():
            return None
        with open(state_file, encoding="utf-8") as f:
            return json.load(f)

    def save_document(self, project_id: str, doc_name: str, content: str) -> Path:
        """Save a document to the docs directory."""
        dirs = self.ensure_project_dirs(project_id)
        doc_path = dirs["docs"] / doc_name
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write(content)
        return doc_path

    def load_document(self, project_id: str, doc_name: str) -> str | None:
        """Load a document from the docs directory."""
        doc_path = self.get_project_path(project_id) / "docs" / doc_name
        if not doc_path.exists():
            return None
        with open(doc_path, encoding="utf-8") as f:
            return f.read()

    def append_log(self, project_id: str, log_entry: dict[str, Any]) -> None:
        """Append a log entry to the logs file."""
        dirs = self.ensure_project_dirs(project_id)
        log_file = dirs["logs"] / "activity.log"
        log_entry["timestamp"] = datetime.now().isoformat()
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    def list_projects(self) -> list[str]:
        """List all project IDs."""
        if not self.base_path.exists():
            return []
        return [p.name for p in self.base_path.iterdir() if p.is_dir()]

    def delete_project(self, project_id: str) -> bool:
        """Delete a project and all its files."""
        import shutil

        project_path = self.get_project_path(project_id)
        if project_path.exists():
            shutil.rmtree(project_path)
            return True
        return False

    def project_exists(self, project_id: str) -> bool:
        """Check if a project exists."""
        return self.get_project_path(project_id).exists()
