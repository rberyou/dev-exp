"""CLI entry point for DevCrew."""

import argparse
import sys

from devcrew import Project, DevCrew
from devcrew.manager.project import ProjectStatus
from devcrew.persistence.file_store import FileStore
from devcrew.ui.web import run_web_ui
from devcrew.ui.tui import run_tui


def cmd_create(args) -> None:
    """Create a new project."""
    store = FileStore(args.store)
    project = Project.create(
        project_id=args.id,
        name=args.name,
        description=args.description or "",
        requirements=args.requirements,
        store=store,
    )
    print(f"✅ 项目创建成功: {project.project_id}")
    print(f"   名称: {project.name}")
    print(f"   状态: {project.status.value}")


def cmd_list(args) -> None:
    """List all projects."""
    store = FileStore(args.store)
    projects = store.list_projects()

    if not projects:
        print("暂无项目")
        return

    print(f"共有 {len(projects)} 个项目:\n")
    for project_id in projects:
        project = Project.load(project_id, store)
        if project:
            progress = project.get_overall_progress()
            print(f"  {project.name} [{progress['status']}] - {progress['tasks']['percentage']}%")


def cmd_start(args) -> None:
    """Start or resume a project."""
    store = FileStore(args.store)
    project = Project.load(args.project_id, store)

    if not project:
        print(f"❌ 项目不存在: {args.project_id}")
        return

    print(f"🚀 启动项目: {project.name}")

    if project.status == ProjectStatus.PAUSED:
        print("   继续执行...")
        project.resume()

    crew = DevCrew(project, model=args.model)
    results = crew.run()

    print(f"\n✅ 项目执行完成!")
    print(f"   最终状态: {project.status.value}")


def cmd_run_phase(args) -> None:
    """Run a specific phase."""
    store = FileStore(args.store)
    project = Project.load(args.project_id, store)

    if not project:
        print(f"❌ 项目不存在: {args.project_id}")
        return

    crew = DevCrew(project, model=args.model)
    results = crew.run_phase(args.phase)

    print(f"✅ 阶段 {args.phase} 完成")
    print(results)


def cmd_status(args) -> None:
    """Show project status."""
    store = FileStore(args.store)
    project = Project.load(args.project_id, store)

    if not project:
        print(f"❌ 项目不存在: {args.project_id}")
        return

    progress = project.get_overall_progress()

    print(f"\n项目: {project.name}")
    print(f"状态: {progress['status']}")
    print(f"阶段: {progress['current_phase']}")
    print(f"进度: {progress['tasks']['percentage']}%")
    print(f"\n任务:")
    for task in project.get_task_tree():
        icon = {"completed": "✅", "in_progress": "🔄", "pending": "⏳", "failed": "❌"}.get(task["status"], "❓")
        print(f"  {icon} {task['title']} [{task['status']}]")


def cmd_web(args) -> None:
    """Start web UI."""
    store = FileStore(args.store)
    print(f"🌐 启动Web界面: http://localhost:{args.port}")
    run_web_ui(host=args.host, port=args.port, store=store)


def cmd_tui(args) -> None:
    """Start TUI."""
    store = FileStore(args.store)
    run_tui(store)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="DevCrew - 多智能体开发团队")
    parser.add_argument("--store", default="./projects", help="项目存储路径")

    subparsers = parser.add_subparsers(dest="command", help="子命令")

    create_parser = subparsers.add_parser("create", help="创建新项目")
    create_parser.add_argument("--id", required=True, help="项目ID")
    create_parser.add_argument("--name", required=True, help="项目名称")
    create_parser.add_argument("--description", help="项目描述")
    create_parser.add_argument("--requirements", required=True, help="需求描述")

    list_parser = subparsers.add_parser("list", help="列出所有项目")

    start_parser = subparsers.add_parser("start", help="启动项目")
    start_parser.add_argument("project_id", help="项目ID")
    start_parser.add_argument("--model", default="gpt-4o", help="使用的模型")

    phase_parser = subparsers.add_parser("phase", help="运行特定阶段")
    phase_parser.add_argument("project_id", help="项目ID")
    phase_parser.add_argument("phase", choices=["requirements", "architecture", "implementation", "testing", "documentation"])
    phase_parser.add_argument("--model", default="gpt-4o", help="使用的模型")

    status_parser = subparsers.add_parser("status", help="查看项目状态")
    status_parser.add_argument("project_id", help="项目ID")

    web_parser = subparsers.add_parser("web", help="启动Web界面")
    web_parser.add_argument("--host", default="0.0.0.0", help="主机")
    web_parser.add_argument("--port", type=int, default=5000, help="端口")

    tui_parser = subparsers.add_parser("tui", help="启动终端界面")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    commands = {
        "create": cmd_create,
        "list": cmd_list,
        "start": cmd_start,
        "phase": cmd_run_phase,
        "status": cmd_status,
        "web": cmd_web,
        "tui": cmd_tui,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
