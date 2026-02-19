"""Task definitions for DevCrew."""

from typing import Any

from crewai import Agent, Task


class DevCrewTasks:
    """Collection of tasks for DevCrew development team."""

    @staticmethod
    def requirements_analysis(
        agent: Agent,
        requirements: str,
        context: str = "",
        **kwargs: Any,
    ) -> Task:
        """Create a requirements analysis task."""
        return Task(
            description=f"""
                分析用户需求，转化为详细的功能列表。

                用户需求：
                {requirements}

                {f'额外上下文：{context}' if context else ''}

                请完成以下工作：
                1. 理解并澄清用户需求
                2. 识别功能需求和非功能需求
                3. 列出详细的 功能列表
                4. 识别潜在的依赖和风险
                5. 提出需要确认的问题
            """,
            expected_output="""
                一份详细的需求分析报告，包括：
                - 需求澄清
                - 功能列表（带优先级）
                - 非功能需求
                - 依赖和风险
                - 待确认问题
            """,
            agent=agent,
            **kwargs,
        )

    @staticmethod
    def architecture_design(
        agent: Agent,
        requirements: str,
        context: str = "",
        **kwargs: Any,
    ) -> Task:
        """Create an architecture design task."""
        return Task(
            description=f"""
                根据需求分析结果，设计技术方案。

                需求分析：
                {requirements}

                {f'额外上下文：{context}' if context else ''}

                请完成以下工作：
                1. 分析技术需求
                2. 选择合适的技术栈
                3. 设计系统架构
                4. 定义模块接口
                5. 制定开发计划
            """,
            expected_output="""
                一份架构设计文档，包括：
                - 技术栈选择理由
                - 系统架构图
                - 模块设计
                - API设计
                - 开发计划
            """,
            agent=agent,
            **kwargs,
        )

    @staticmethod
    def implementation(
        agent: Agent,
        architecture: str,
        context: str = "",
        **kwargs: Any,
    ) -> Task:
        """Create an implementation task."""
        return Task(
            description=f"""
                根据架构设计文档实现代码。

                架构设计：
                {architecture}

                {f'额外上下文：{context}' if context else ''}

                请完成以下工作：
                1. 创建项目结构
                2. 实现核心模块
                3. 编写单元测试
                4. 确保代码质量
                5. 更新相关文档
            """,
            expected_output="""
                实现完成的代码，包括：
                - 项目结构
                - 核心代码文件
                - 单元测试
                - 更新后的文档
            """,
            agent=agent,
            **kwargs,
        )

    @staticmethod
    def testing(
        agent: Agent,
        implementation: str,
        requirements: str,
        context: str = "",
        **kwargs: Any,
    ) -> Task:
        """Create a testing task."""
        return Task(
            description=f"""
                对实现的功能进行全面测试。

                实现内容：
                {implementation}

                原始需求：
                {requirements}

                {f'额外上下文：{context}' if context else ''}

                请完成以下工作：
                1. 编写测试用例
                2. 执行单元测试
                3. 执行集成测试
                4. 执行端到端测试
                5. 提交测试报告
            """,
            expected_output="""
                测试报告，包括：
                - 测试用例列表
                - 测试结果
                - 覆盖率报告
                - 问题列表
            """,
            agent=agent,
            **kwargs,
        )

    @staticmethod
    def documentation(
        agent: Agent,
        context: str = "",
        **kwargs: Any,
    ) -> Task:
        """Create a documentation task."""
        return Task(
            description=f"""
                整理项目文档。

                {f'项目上下文：{context}' if context else ''}

                请完成以下工作：
                1. 整理需求文档
                2. 整理架构文档
                3. 整理API文档
                4. 整理部署文档
                5. 创建README
            """,
            expected_output="""
                完整的项目文档，包括：
                - README.md
                - API文档
                - 部署文档
                - 开发者指南
            """,
            agent=agent,
            **kwargs,
        )

    @staticmethod
    def project_coordination(
        agent: Agent,
        project_context: str,
        **kwargs: Any,
    ) -> Task:
        """Create a project coordination task."""
        return Task(
            description=f"""
                协调整个开发团队的工作。

                项目状态：
                {project_context}

                请完成以下工作：
                1. 评估当前进度
                2. 识别阻塞问题
                3. 分配任务给各个Agent
                4. 监控质量
                5. 汇报项目状态
            """,
            expected_output="""
                项目协调报告，包括：
                - 当前进度
                - 问题列表
                - 下一步计划
            """,
            agent=agent,
            **kwargs,
        )
