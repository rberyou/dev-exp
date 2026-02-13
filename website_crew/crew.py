from crewai import Agent, Task, Crew, Process, LLM
from pathlib import Path
import yaml
import os

from tools.code_writer import write_code_file, create_directory, read_code_file, list_output_files
from tools.file_manager import init_nextjs_project, create_config_file, get_project_stats


def get_llm_from_env():
    """Create LLM instance from environment variables."""
    model = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE")
    
    llm_kwargs = {"model": model}
    
    if api_key:
        llm_kwargs["api_key"] = api_key
    if base_url:
        llm_kwargs["base_url"] = base_url
    
    return LLM(**llm_kwargs)


class WebsiteDevCrew:
    """Website Development Crew - A team of AI agents that build websites"""
    
    def __init__(self):
        config_path = Path(__file__).parent / 'config'
        
        with open(config_path / 'agents.yaml', 'r', encoding='utf-8') as f:
            self.agents_config = yaml.safe_load(f)
        with open(config_path / 'tasks.yaml', 'r', encoding='utf-8') as f:
            self.tasks_config = yaml.safe_load(f)
        
        self.tools = [
            write_code_file, create_directory, read_code_file, list_output_files,
            init_nextjs_project, create_config_file, get_project_stats
        ]
        
        self.llm = get_llm_from_env()
        self._agents = None
        self._tasks = None

    def _create_agent(self, config_key: str) -> Agent:
        """Helper method to create an agent with common configuration."""
        config = self.agents_config[config_key]
        return Agent(
            role=config['role'],
            goal=config['goal'],
            backstory=config['backstory'],
            verbose=config.get('verbose', True),
            tools=self.tools,
            llm=self.llm
        )

    def _create_agents(self):
        if self._agents is None:
            self._agents = [
                self._create_agent('product_manager'),
                self._create_agent('ui_designer'),
                self._create_agent('frontend_dev'),
                self._create_agent('backend_dev'),
                self._create_agent('qa_engineer')
            ]
        return self._agents

    def _create_tasks(self, agents):
        pm, ui, fe, be, qa = agents
        
        analyze_config = self.tasks_config['analyze_requirements']
        design_config = self.tasks_config['design_system']
        frontend_config = self.tasks_config['implement_frontend']
        backend_config = self.tasks_config['implement_backend']
        test_config = self.tasks_config['write_tests']
        
        analyze_task = Task(
            description=analyze_config['description'],
            expected_output=analyze_config['expected_output'],
            agent=pm
        )
        
        design_task = Task(
            description=design_config['description'],
            expected_output=design_config['expected_output'],
            agent=ui,
            context=[analyze_task]
        )
        
        frontend_task = Task(
            description=frontend_config['description'],
            expected_output=frontend_config['expected_output'],
            agent=fe,
            context=[design_task]
        )
        
        backend_task = Task(
            description=backend_config['description'],
            expected_output=backend_config['expected_output'],
            agent=be,
            context=[analyze_task]
        )
        
        test_task = Task(
            description=test_config['description'],
            expected_output=test_config['expected_output'],
            agent=qa,
            context=[frontend_task, backend_task]
        )
        
        return [analyze_task, design_task, frontend_task, backend_task, test_task]

    def crew(self) -> Crew:
        agents = self._create_agents()
        tasks = self._create_tasks(agents)
        
        return Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=True
        )
