"""
Code RAG System - LLM 集成
支持 vLLM、llama.cpp、Ollama 等本地部署方案
"""

import os
import json
from typing import List, Dict, Optional, Generator, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import requests


@dataclass
class LLMConfig:
    """LLM配置"""
    # API配置
    api_base: str = "http://localhost:8000/v1"
    api_key: str = "not-needed"  # 本地部署通常不需要
    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    
    # 生成参数
    max_tokens: int = 2048
    temperature: float = 0.1
    top_p: float = 0.95
    
    # 上下文管理
    max_context_tokens: int = 6000
    reserved_output_tokens: int = 1024
    
    # 超时设置
    timeout: int = 120


class BaseLLMClient(ABC):
    """LLM客户端基类"""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """生成响应"""
        pass
    
    @abstractmethod
    def generate_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """流式生成响应"""
        pass
    
    @abstractmethod
    def chat(self, messages: List[Dict], **kwargs) -> str:
        """对话式生成"""
        pass


class OpenAICompatibleClient(BaseLLMClient):
    """
    OpenAI兼容的API客户端
    支持 vLLM、text-generation-webui、LocalAI 等
    """
    
    def __init__(self, config: LLMConfig = None):
        self.config = config or LLMConfig()
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}"
        })
    
    def _build_messages(self, prompt: str, system_prompt: str = None) -> List[Dict]:
        """构建消息列表"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages
    
    def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        max_tokens: int = None,
        temperature: float = None,
        stop: List[str] = None,
        **kwargs
    ) -> str:
        """
        生成响应
        
        Args:
            prompt: 用户提示
            system_prompt: 系统提示
            max_tokens: 最大token数
            temperature: 温度
            stop: 停止词列表
        
        Returns:
            生成的文本
        """
        messages = self._build_messages(prompt, system_prompt)
        return self.chat(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            **kwargs
        )
    
    def generate_stream(
        self,
        prompt: str,
        system_prompt: str = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """流式生成"""
        messages = self._build_messages(prompt, system_prompt)
        yield from self.chat_stream(messages, **kwargs)
    
    def chat(
        self,
        messages: List[Dict],
        max_tokens: int = None,
        temperature: float = None,
        stop: List[str] = None,
        **kwargs
    ) -> str:
        """
        对话式生成
        
        Args:
            messages: 消息列表
            max_tokens: 最大token数
            temperature: 温度
            stop: 停止词
        
        Returns:
            生成的文本
        """
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "max_tokens": max_tokens or self.config.max_tokens,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "top_p": self.config.top_p,
        }
        
        if stop:
            payload["stop"] = stop
        
        try:
            response = self.session.post(
                f"{self.config.api_base}/chat/completions",
                json=payload,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            return f"Error: {str(e)}"
    
    def chat_stream(
        self,
        messages: List[Dict],
        max_tokens: int = None,
        temperature: float = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """流式对话"""
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "max_tokens": max_tokens or self.config.max_tokens,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "stream": True,
        }
        
        try:
            response = self.session.post(
                f"{self.config.api_base}/chat/completions",
                json=payload,
                timeout=self.config.timeout,
                stream=True
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            content = chunk["choices"][0]["delta"].get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue
        except requests.exceptions.RequestException as e:
            yield f"Error: {str(e)}"
    
    def count_tokens(self, text: str) -> int:
        """估算token数量"""
        # 简单估算：中文约1.5字符/token，英文约4字符/token
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        other_chars = len(text) - chinese_chars
        return int(chinese_chars / 1.5 + other_chars / 4)


class OllamaClient(BaseLLMClient):
    """
    Ollama 客户端
    适用于 Ollama 本地部署
    """
    
    def __init__(self, config: LLMConfig = None):
        self.config = config or LLMConfig(api_base="http://localhost:11434")
        self.session = requests.Session()
    
    def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        max_tokens: int = None,
        temperature: float = None,
        **kwargs
    ) -> str:
        """生成响应"""
        payload = {
            "model": self.config.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens or self.config.max_tokens,
                "temperature": temperature if temperature is not None else self.config.temperature,
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            response = self.session.post(
                f"{self.config.api_base}/api/generate",
                json=payload,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            return response.json()["response"]
        except requests.exceptions.RequestException as e:
            return f"Error: {str(e)}"
    
    def generate_stream(
        self,
        prompt: str,
        system_prompt: str = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """流式生成"""
        payload = {
            "model": self.config.model_name,
            "prompt": prompt,
            "stream": True,
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            response = self.session.post(
                f"{self.config.api_base}/api/generate",
                json=payload,
                timeout=self.config.timeout,
                stream=True
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if chunk.get("response"):
                        yield chunk["response"]
                    if chunk.get("done"):
                        break
        except requests.exceptions.RequestException as e:
            yield f"Error: {str(e)}"
    
    def chat(self, messages: List[Dict], **kwargs) -> str:
        """对话式生成"""
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "stream": False,
        }
        
        try:
            response = self.session.post(
                f"{self.config.api_base}/api/chat",
                json=payload,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            return response.json()["message"]["content"]
        except requests.exceptions.RequestException as e:
            return f"Error: {str(e)}"


class CodeRAGPrompts:
    """
    代码RAG专用提示模板
    """
    
    SYSTEM_PROMPT = """你是一个专业的代码助手，专门帮助开发者理解和维护代码库。
你的任务是基于提供的代码上下文，准确回答用户关于代码的问题。

回答原则：
1. 基于提供的代码上下文进行回答，不要编造不存在的代码
2. 如果上下文不足以回答问题，请明确说明
3. 使用清晰的技术语言，必要时提供代码示例
4. 指出代码中的潜在问题或改进建议"""
    
    CODE_QA_TEMPLATE = """## 相关代码上下文

{context}

## 用户问题

{question}

## 回答要求

请基于上述代码上下文回答用户的问题。如果需要引用代码，请注明文件路径和行号。"""
    
    CODE_EXPLAIN_TEMPLATE = """## 代码片段

文件：{file_path}
行号：{start_line}-{end_line}

```{language}
{code}
```

## 任务

请详细解释这段代码的功能、实现逻辑和关键点。包括：
1. 代码的主要功能
2. 关键变量和数据结构
3. 算法或逻辑流程
4. 可能的改进点或潜在问题"""
    
    CODE_FIX_TEMPLATE = """## 问题描述

{problem}

## 相关代码

{context}

## 任务

请分析问题原因，并提供修复方案。包括：
1. 问题根因分析
2. 修复代码
3. 修复说明"""
    
    CODE_REVIEW_TEMPLATE = """## 代码变更

{diff}

## 相关上下文

{context}

## 任务

请对这段代码进行审查，关注：
1. 代码正确性
2. 性能问题
3. 安全隐患
4. 代码风格和可维护性
5. 改进建议"""
    
    @classmethod
    def format_code_qa(cls, context: str, question: str) -> str:
        """格式化代码问答提示"""
        return cls.CODE_QA_TEMPLATE.format(context=context, question=question)
    
    @classmethod
    def format_code_explain(
        cls,
        code: str,
        file_path: str,
        language: str,
        start_line: int,
        end_line: int
    ) -> str:
        """格式化代码解释提示"""
        return cls.CODE_EXPLAIN_TEMPLATE.format(
            code=code,
            file_path=file_path,
            language=language,
            start_line=start_line,
            end_line=end_line
        )


class LLMManager:
    """
    LLM管理器
    统一管理不同的LLM后端
    """
    
    BACKENDS = {
        "openai": OpenAICompatibleClient,
        "vllm": OpenAICompatibleClient,
        "ollama": OllamaClient,
    }
    
    def __init__(self, config: LLMConfig = None, backend: str = "openai"):
        self.config = config or LLMConfig()
        self.backend = backend
        self.client = self.BACKENDS[backend](config)
        self.prompts = CodeRAGPrompts()
    
    def generate(self, prompt: str, **kwargs) -> str:
        """生成响应"""
        return self.client.generate(prompt, **kwargs)
    
    def generate_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """流式生成"""
        yield from self.client.generate_stream(prompt, **kwargs)
    
    def chat(self, messages: List[Dict], **kwargs) -> str:
        """对话生成"""
        return self.client.chat(messages, **kwargs)
    
    def answer_code_question(
        self,
        question: str,
        code_context: List[Dict],
        stream: bool = False
    ) -> Union[str, Generator[str, None, None]]:
        """
        回答代码问题
        
        Args:
            question: 用户问题
            code_context: 代码上下文列表
            stream: 是否流式输出
        
        Returns:
            回答文本或生成器
        """
        # 格式化上下文
        context_parts = []
        for ctx in code_context:
            header = f"### {ctx.get('metadata', {}).get('file_path', 'Unknown')}"
            if ctx.get("metadata", {}).get("symbol_name"):
                header += f" - {ctx['metadata']['symbol_name']}"
            lines = f"(Lines {ctx.get('metadata', {}).get('start_line', '?')}-{ctx.get('metadata', {}).get('end_line', '?')})"
            
            context_parts.append(f"{header} {lines}\n```\n{ctx.get('content', '')}\n```")
        
        context = "\n\n".join(context_parts)
        
        # 检查上下文长度
        total_tokens = self.client.count_tokens(context + question)
        if total_tokens > self.config.max_context_tokens:
            # 截断上下文
            context = self._truncate_context(context_parts, question)
        
        prompt = self.prompts.format_code_qa(context, question)
        
        if stream:
            return self.client.generate_stream(
                prompt,
                system_prompt=self.prompts.SYSTEM_PROMPT
            )
        else:
            return self.client.generate(
                prompt,
                system_prompt=self.prompts.SYSTEM_PROMPT
            )
    
    def _truncate_context(self, context_parts: List[str], question: str) -> str:
        """智能截断上下文"""
        max_tokens = self.config.max_context_tokens - self.config.reserved_output_tokens
        question_tokens = self.client.count_tokens(question)
        available_tokens = max_tokens - question_tokens - 500  # 预留prompt模板
        
        selected_parts = []
        current_tokens = 0
        
        for part in context_parts:
            part_tokens = self.client.count_tokens(part)
            if current_tokens + part_tokens <= available_tokens:
                selected_parts.append(part)
                current_tokens += part_tokens
            else:
                break
        
        return "\n\n".join(selected_parts)


# 使用示例
if __name__ == "__main__":
    # 创建LLM客户端
    config = LLMConfig(
        api_base="http://localhost:8000/v1",
        model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
        max_tokens=2048,
        temperature=0.1
    )
    
    manager = LLMManager(config, backend="openai")
    
    # 测试生成
    # response = manager.generate("解释什么是快速排序算法")
    # print(response)
    
    # 测试流式生成
    # for chunk in manager.generate_stream("解释什么是快速排序算法"):
    #     print(chunk, end="", flush=True)
