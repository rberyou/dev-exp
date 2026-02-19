"""Web UI for DevCrew."""

import json
from flask import Flask, render_template, jsonify, request

from devcrew.manager.project import Project
from devcrew.manager.task_tracker import TaskStatus
from devcrew.persistence.file_store import FileStore


def create_app(store: FileStore | None = None) -> Flask:
    """Create and configure the Flask app."""
    app = Flask(__name__, template_folder="../../templates")
    app.store = store or FileStore()

    @app.route("/")
    def index():
        """Dashboard page showing all projects."""
        projects = app.store.list_projects()
        project_list = []
        for project_id in projects:
            project = Project.load(project_id, app.store)
            if project:
                project_list.append(project.get_overall_progress())
        return render_template("dashboard.html", projects=project_list)

    @app.route("/project/<project_id>")
    def project_detail(project_id: str):
        """Project detail page."""
        project = Project.load(project_id, app.store)
        if not project:
            return "Project not found", 404

        progress = project.get_overall_progress()
        tasks = project.get_task_tree()
        return render_template(
            "project.html",
            project=project,
            progress=progress,
            tasks=tasks,
        )

    @app.route("/api/projects", methods=["GET"])
    def api_list_projects():
        """API: List all projects."""
        projects = app.store.list_projects()
        project_list = []
        for project_id in projects:
            project = Project.load(project_id, app.store)
            if project:
                project_list.append(project.get_overall_progress())
        return jsonify(project_list)

    @app.route("/api/projects", methods=["POST"])
    def api_create_project():
        """API: Create a new project."""
        data = request.json
        project_id = data.get("project_id", f"project_{len(app.store.list_projects()) + 1}")
        name = data.get("name", "New Project")
        description = data.get("description", "")
        requirements = data.get("requirements", "")

        project = Project.create(
            project_id=project_id,
            name=name,
            description=description,
            requirements=requirements,
            store=app.store,
        )
        return jsonify(project.to_dict())

    @app.route("/api/projects/<project_id>", methods=["GET"])
    def api_get_project(project_id: str):
        """API: Get project details."""
        project = Project.load(project_id, app.store)
        if not project:
            return jsonify({"error": "Project not found"}), 404
        return jsonify(project.to_dict())

    @app.route("/api/projects/<project_id>/progress", methods=["GET"])
    def api_get_progress(project_id: str):
        """API: Get project progress."""
        project = Project.load(project_id, app.store)
        if not project:
            return jsonify({"error": "Project not found"}), 404
        return jsonify(project.get_overall_progress())

    @app.route("/api/projects/<project_id>/tasks", methods=["GET"])
    def api_get_tasks(project_id: str):
        """API: Get all tasks for a project."""
        project = Project.load(project_id, app.store)
        if not project:
            return jsonify({"error": "Project not found"}), 404
        return jsonify(project.get_task_tree())

    @app.route("/api/projects/<project_id>/start", methods=["POST"])
    def api_start_project(project_id: str):
        """API: Start or resume project execution."""
        project = Project.load(project_id, app.store)
        if not project:
            return jsonify({"error": "Project not found"}), 404

        from devcrew.crew.crew import DevCrew

        crew = DevCrew(project)
        result = crew.run()
        return jsonify({"status": "completed", "result": str(result)})

    @app.route("/api/projects/<project_id>/phase/<phase>", methods=["POST"])
    def api_run_phase(project_id: str, phase: str):
        """API: Run a specific phase."""
        project = Project.load(project_id, app.store)
        if not project:
            return jsonify({"error": "Project not found"}), 404

        from devcrew.crew.crew import DevCrew

        crew = DevCrew(project)
        result = crew.run_phase(phase)
        return jsonify({"status": "completed", "phase": phase, "result": str(result)})

    @app.route("/api/projects/<project_id>/pause", methods=["POST"])
    def api_pause_project(project_id: str):
        """API: Pause project execution."""
        project = Project.load(project_id, app.store)
        if not project:
            return jsonify({"error": "Project not found"}), 404
        project.pause()
        return jsonify({"status": "paused"})

    @app.route("/api/projects/<project_id>/resume", methods=["POST"])
    def api_resume_project(project_id: str):
        """API: Resume project execution."""
        project = Project.load(project_id, app.store)
        if not project:
            return jsonify({"error": "Project not found"}), 404

        from devcrew.crew.crew import DevCrew
        from devcrew.manager.project import ProjectStatus

        crew = DevCrew(project)
        project.status = ProjectStatus.IN_PROGRESS
        project.save_state()
        result = crew.run()
        return jsonify({"status": "resumed", "result": str(result)})

    return app


def run_web_ui(host: str = "0.0.0.0", port: int = 5000, store: FileStore | None = None):
    """Run the web UI server."""
    app = create_app(store)
    app.run(host=host, port=port, debug=True)


if __name__ == "__main__":
    run_web_ui()
