"""Crew orchestration for DevCrew."""

from typing import Any

from crewai import Crew, LLM, Process, Task

from devcrew.crew.agents import DevCrewAgents
from devcrew.crew.tasks import DevCrewTasks
from devcrew.manager.project import Project, ProjectStatus
from devcrew.manager.task_tracker import TaskStatus


class DevCrew:
    """Multi-Agent Development Team using CrewAI."""

    def __init__(
        self,
        project: Project,
        llm: str | LLM | None = None,
        model: str = "gpt-4o",
        **kwargs: Any,
    ):
        """Initialize DevCrew.

        Args:
            project: Project instance to manage
            llm: LLM instance or model string
            model: Default model to use if llm not provided
        """
        self.project = project
        self.llm = llm or LLM(model=model)
        self.kwargs = kwargs

        self.agents = DevCrewAgents()
        self.tasks = DevCrewTasks()

    def _create_agents(self) -> dict[str, Any]:
        """Create all agents for the crew."""
        return {
            "requirements_analyst": self.agents.requirements_analyst(llm=self.llm),
            "architect": self.agents.architect(llm=self.llm),
            "developer": self.agents.developer(llm=self.llm),
            "tester": self.agents.tester(llm=self.llm),
            "documenter": self.agents.documenter(llm=self.llm),
            "manager": self.agents.manager(llm=self.llm),
        }

    def _run_phase(self, phase: str) -> dict[str, Any]:
        """Run a specific phase of development."""
        agents = self._create_agents()
        results = {}

        if phase == "requirements":
            task = self.tasks.requirements_analysis(
                agent=agents["requirements_analyst"],
                requirements=self.project.requirements,
            )
            crew = Crew(
                agents=[agents["requirements_analyst"]],
                tasks=[task],
                process=Process.sequential,
                verbose=True,
            )
            result = crew.kickoff()
            results["requirements"] = result.raw

            task_obj = self.project.task_tracker.get_task("req_analysis")
            if task_obj:
                task_obj.update_status(TaskStatus.COMPLETED, output=str(result.raw))
            self.project.save_state()

        elif phase == "architecture":
            task = self.tasks.architecture_design(
                agent=agents["architect"],
                requirements=results.get("requirements", ""),
            )
            crew = Crew(
                agents=[agents["architect"]],
                tasks=[task],
                process=Process.sequential,
                verbose=True,
            )
            result = crew.kickoff()
            results["architecture"] = result.raw

            task_obj = self.project.task_tracker.get_task("architecture")
            if task_obj:
                task_obj.update_status(TaskStatus.COMPLETED, output=str(result.raw))
            self.project.save_state()

        elif phase == "implementation":
            task = self.tasks.implementation(
                agent=agents["developer"],
                architecture=results.get("architecture", ""),
            )
            crew = Crew(
                agents=[agents["developer"]],
                tasks=[task],
                process=Process.sequential,
                verbose=True,
            )
            result = crew.kickoff()
            results["implementation"] = result.raw

            task_obj = self.project.task_tracker.get_task("implementation")
            if task_obj:
                task_obj.update_status(TaskStatus.COMPLETED, output=str(result.raw))
            self.project.save_state()

        elif phase == "testing":
            task = self.tasks.testing(
                agent=agents["tester"],
                implementation=results.get("implementation", ""),
                requirements=self.project.requirements,
            )
            crew = Crew(
                agents=[agents["tester"]],
                tasks=[task],
                process=Process.sequential,
                verbose=True,
            )
            result = crew.kickoff()
            results["testing"] = result.raw

            task_obj = self.project.task_tracker.get_task("testing")
            if task_obj:
                task_obj.update_status(TaskStatus.COMPLETED, output=str(result.raw))
            self.project.save_state()

        elif phase == "documentation":
            task = self.tasks.documentation(
                agent=agents["documenter"],
                context=str(results),
            )
            crew = Crew(
                agents=[agents["documenter"]],
                tasks=[task],
                process=Process.sequential,
                verbose=True,
            )
            result = crew.kickoff()
            results["documentation"] = result.raw

            task_obj = self.project.task_tracker.get_task("documentation")
            if task_obj:
                task_obj.update_status(TaskStatus.COMPLETED, output=str(result.raw))
            self.project.save_state()

        return results

    def run(self, start_phase: str | None = None) -> dict[str, Any]:
        """Run the development process.

        Args:
            start_phase: Optional phase to start from (for resuming)

        Returns:
            Dictionary containing results from all phases
        """
        phases = ["requirements", "architecture", "implementation", "testing", "documentation"]

        if start_phase and start_phase in phases:
            start_idx = phases.index(start_phase)
            phases = phases[start_idx:]

        self.project.status = ProjectStatus.IN_PROGRESS
        self.project.save_state()

        all_results = {}
        for phase in phases:
            self.project.current_phase = phase
            self.project.save_state()

            phase_results = self._run_phase(phase)
            all_results.update(phase_results)

        self.project.complete()
        return all_results

    def run_phase(self, phase: str) -> dict[str, Any]:
        """Run a specific phase.

        Args:
            phase: Phase to run (requirements, architecture, implementation, testing, documentation)

        Returns:
            Results from the phase
        """
        return self._run_phase(phase)

    def get_status(self) -> dict[str, Any]:
        """Get current project status."""
        return self.project.to_dict()
