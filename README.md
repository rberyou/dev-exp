# 工作空间项目管理

此仓库采用多分支策略管理多个项目，每个项目在独立的分支上开发。

## 分支说明

- `main` - 主分支，包含工作空间的基本配置
- `code-rag-system-branch` - code-rag-system 项目的开发分支

## 项目分支操作指南

### 查看所有分支
```bash
git branch -a
```

### 切换到特定项目分支
```bash
git checkout <branch-name>
```

### 在项目分支上工作
```bash
# 切换到项目分支
git checkout code-rag-system-branch

# 进行修改
# ...

# 提交更改
git add .
git commit -m "描述你的更改"
git push origin code-rag-system-branch
```

### 创建新的项目分支
如果要在工作空间中添加新项目：
```bash
git checkout main
git checkout -b <new-project-branch-name>
# 添加新项目文件
git add .
git commit -m "Add <new-project-name>"
```