"""Terminal UI for DevCrew using Textual."""

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Button, Header, Footer, Label, ListView, ListItem, ProgressBar, Static
from textual import work

from devcrew.manager.project import Project, ProjectStatus
from devcrew.manager.task_tracker import TaskStatus
from devcrew.persistence.file_store import FileStore


class ProjectListScreen(App):
    """Main screen showing project list."""

    CSS = """
    Screen {
        background: $surface;
    }
    #sidebar {
        width: 30;
        background: $panel;
        border-right: solid $border;
    }
    #main {
        width: 1fr;
    }
    .project-item {
        height: 3;
        padding: 1;
    }
    .project-item:hover {
        background: $hover;
    }
    """

    BINDINGS = [
        ("q", "quit", "退出"),
        ("n", "new_project", "新建项目"),
        ("r", "refresh", "刷新"),
    ]

    def __init__(self, store: FileStore | None = None):
        super().__init__()
        self.store = store or FileStore()
        self.projects = []

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Vertical(id="sidebar"):
                yield Label("[bold]DevCrew[/bold]\n多智能体开发团队", id="title")
                yield Button("+ 新建项目", id="new_project", variant="primary")
                yield Button("🔄 刷新", id="refresh")
            with Vertical(id="main"):
                yield Label("项目列表", id="header")
                yield ListView(id="project_list")
        yield Footer()

    def on_mount(self) -> None:
        self.load_projects()

    def load_projects(self) -> None:
        """Load projects from storage."""
        self.projects = []
        for project_id in self.store.list_projects():
            project = Project.load(project_id, self.store)
            if project:
                self.projects.append(project)
        self.update_list()

    def update_list(self) -> None:
        """Update the project list view."""
        list_view = self.query_one("#project_list", ListView)
        list_view.clear()
        for project in self.projects:
            progress = project.get_overall_progress()
            label = f"{project.name} [{progress['status']}] {progress['tasks']['percentage']}%"
            list_view.append(ListItem(Label(label), id=project.project_id))

    def action_new_project(self) -> None:
        """Show new project dialog."""
        self.push_screen(NewProjectDialog(self.store), self.on_project_created)

    def on_project_created(self, project: Project | None) -> None:
        """Handle project creation."""
        if project:
            self.load_projects()

    def action_refresh(self) -> None:
        """Refresh project list."""
        self.load_projects()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle project selection."""
        if event.item:
            project_id = event.item.id
            if project_id:
                self.push_screen(ProjectDetailScreen(project_id, self.store), self.on_detail_closed)

    def on_detail_closed(self) -> None:
        """Handle detail screen close."""
        self.load_projects()


class NewProjectDialog(App):
    """Dialog for creating a new project."""

    CSS = """
    Dialog {
        width: 60;
        height: auto;
    }
    #form {
        padding: 1;
    }
    .field {
        margin-bottom: 1;
    }
    Label {
        display: block;
        margin-bottom: 0;
    }
    Input {
        width: 100%;
    }
    """

    def __init__(self, store: FileStore):
        super().__init__()
        self.store = store
        self.project: Project | None = None

    def compose(self) -> ComposeResult:
        yield Container(
            Vertical(
                Label("[bold]新建项目[/bold]"),
                Label("项目名称:"),
                Input(id="name"),
                Label("项目描述:"),
                Input(id="description"),
                Label("需求描述:"),
                Input(id="requirements"),
                Horizontal(
                    Button("取消", id="cancel", variant="default"),
                    Button("创建", id="create", variant="primary"),
                ),
                id="form",
            )
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press."""
        if event.button.id == "cancel":
            self.dismiss(None)
        elif event.button.id == "create":
            name = self.query_one("#name", Input).value
            description = self.query_one("#description", Input).value
            requirements = self.query_one("#requirements", Input).value

            if not name or not requirements:
                return

            project_id = f"project_{len(self.store.list_projects()) + 1}"
            self.project = Project.create(
                project_id=project_id,
                name=name,
                description=description,
                requirements=requirements,
                store=self.store,
            )
            self.dismiss(self.project)


class ProjectDetailScreen(App):
    """Screen showing project details."""

    CSS = """
    #header {
        height: auto;
        padding: 1;
        background: $accent;
        color: $text;
    }
    #content {
        padding: 1;
    }
    .stat {
        width: 25%;
        text-align: center;
    }
    .task-row {
        height: auto;
        padding: 1;
        border-bottom: solid $border;
    }
    """

    BINDINGS = [
        ("escape", "pop_screen", "返回"),
        ("s", "start", "开始"),
        ("p", "pause", "暂停"),
    ]

    def __init__(self, project_id: str, store: FileStore):
        super().__init__()
        self.project_id = project_id
        self.store = store
        self.project = Project.load(project_id, store)

    def compose(self) -> ComposeResult:
        if self.project:
            progress = self.project.get_overall_progress()
            yield Container(
                Label(f"[bold]{self.project.name}[/bold] - {progress['status']}", id="header"),
                Container(
                    Label(f"进度: {progress['tasks']['percentage']}%"),
                    Label(f"已完成: {progress['tasks']['completed']}"),
                    Label(f"进行中: {progress['tasks']['in_progress']}"),
                    Label(f"总任务: {progress['tasks']['total']}"),
                    id="stats",
                ),
                Label("[bold]任务列表:[/bold]"),
                Vertical(id="tasks"),
                Horizontal(
                    Button("▶ 开始", id="start", variant="success"),
                    Button("⏸ 暂停", id="pause", variant="warning"),
                    Button("🔄 刷新", id="refresh"),
                ),
            )

    def on_mount(self) -> None:
        self.refresh_tasks()

    def refresh_tasks(self) -> None:
        """Refresh task list."""
        tasks_container = self.query_one("#tasks", Vertical)
        tasks_container.remove_children()

        if self.project:
            for task in self.project.get_task_tree():
                status_icon = {
                    TaskStatus.COMPLETED.value: "✅",
                    TaskStatus.IN_PROGRESS.value: "🔄",
                    TaskStatus.PENDING.value: "⏳",
                    TaskStatus.FAILED.value: "❌",
                }.get(task["status"], "❓")

                label = f"{status_icon} {task['title']} [{task['status']}]"
                tasks_container.add(Static(label, classes="task-row"))

    def action_start(self) -> None:
        """Start project execution."""
        if self.project:
            self.project.status = ProjectStatus.IN_PROGRESS
            self.project.save_state()
            self.refresh_tasks()

    def action_pause(self) -> None:
        """Pause project execution."""
        if self.project:
            self.project.pause()
            self.refresh_tasks()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press."""
        if event.button.id == "start":
            self.action_start()
        elif event.button.id == "pause":
            self.action_pause()
        elif event.button.id == "refresh":
            self.project = Project.load(self.project_id, self.store)
            self.refresh_tasks()


def run_tui(store: FileStore | None = None) -> None:
    """Run the TUI."""
    store = store or FileStore()
    app = ProjectListScreen(store)
    app.run()


if __name__ == "__main__":
    run_tui()
