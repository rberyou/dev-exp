# DevExp - Multi-Agent Collaboration System

> Multi-Agent Project Development System Design

## Overview

This project implements a collaborative workflow system with multiple AI agents working together to develop products efficiently.

## Agent Roles

| Role | Core Responsibility |
|------|---------------------|
| **Product Designer** | Requirements analysis, PRD creation |
| **UI Designer** | Interface & interaction design |
| **Program Architect** | Technical architecture, API definition |
| **Developer** | Code implementation |
| **QA Engineer** | Test case design, quality assurance |
| **Code Reviewer** | Code review, standards compliance |
| **Product Manager** | Project management, coordination |

## Workflow

```
User Request → PRD → UI/Architecture (parallel) → Development → Code Review → QA Testing → User Acceptance
```

## Key Features

- **Parallel Execution**: UI Design and Architecture Design can run in parallel
- **Clear Dependencies**: Well-defined task dependencies and handoff rules
- **Status Tracking**: Bitable-based task Kanban for real-time status updates
- **Review Gates**: Code review before QA testing
- **User Confirmation**: Confirmation nodes at PRD, UI, and final acceptance stages

## Documentation

- [Multi-Agent System Design](./docs/multi-agent-system-design.md) - Detailed design documentation

## Version

- **Current Version**: v1.1
- **Last Updated**: 2026-03-08
