"""监管智能体（Supervisor）——异步子智能体示例。

一个交互式 REPL，演示对 server.py 中 FastAPI 服务发起的五个异步子智能体操作。

监管智能体通过 Agent Protocol（经由 LangGraph SDK）将研究任务委派给
服务器上托管的研究智能体。任务在后台运行——监管智能体立即返回
任务 ID，你可以在准备就绪时查看状态。

运行（在另一个终端启动 server.py 后）：
    uv run python supervisor.py

试试这些提示词：
    > 研究量子计算的最新发展
    > 查看 <task-id> 的状态
    > 更新 <task-id>，使其只关注商业应用
    > 取消 <task-id>
    > 列出所有任务
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from deepagents import create_deep_agent
from deepagents.middleware.async_subagents import AsyncSubAgent

load_dotenv(Path(__file__).parent / ".env")

import os  # noqa: E402

RESEARCHER_URL = os.environ.get("RESEARCHER_URL") or "http://localhost:2024"

# ── 智能体配置 ───────────────────────────────────────────────────────────────

async_subagents: list[AsyncSubAgent] = [
    {
        "name": "researcher",
        "description": (
            "一名使用网络搜索调查任意话题的研究智能体。"
            "在后台运行，返回详细摘要。"
        ),
        "graph_id": "researcher",
        "url": RESEARCHER_URL,
        "headers": {"x-auth-scheme": "custom"},
    },
]

checkpointer = MemorySaver()
thread_id = str(uuid.uuid4())

supervisor = create_deep_agent(
    model=ChatOpenAI(
        model=os.getenv("LLM_MODEL_NAME"),
        base_url=os.getenv("LLM_BASE_URL"),
        api_key=os.getenv("LLM_API_KEY"),
    ),
    checkpointer=checkpointer,
    system_prompt=(
        "你是一名研究监管者，负责协调后台的研究智能体。\n\n"
        "对于一般性问题，直接回答——不要启动研究员。\n\n"
        '只有在用户说出"research"、"investigate"、"look into"或"find out"，'
        '或用中文说出"研究"、"调查"、"查一下"、"了解一下"时，才启动研究员。\n\n'
        "START：当用户要求研究某件事时：\n"
        '  1. 使用 subagent_type "researcher" 和话题调用 start_async_task。\n'
        "  2. 报告 task_id 并停止。不要立即检查状态。\n\n"
        "CHECK：当用户询问状态或结果时：\n"
        "  1. 使用准确的 task_id 调用 check_async_task。\n"
        "  2. 报告工具的返回值。如果还在运行，如实告知并停止。\n\n"
        "UPDATE：当用户要求更改研究员的工作内容时：\n"
        "  1. 使用 task_id 和新指令调用 update_async_task。\n"
        "  2. 确认更新。\n\n"
        "CANCEL：当用户要求取消任务时：\n"
        "  1. 使用准确的 task_id 调用 cancel_async_task。\n"
        "  2. 确认取消。\n\n"
        "LIST：当用户要求列出任务或检查所有状态时：\n"
        "  1. 调用 list_async_tasks。\n"
        "  2. 呈现实时状态。\n\n"
        "规则：\n"
        "- 切勿从内存中报告过期状态。始终调用工具。\n"
        "- 切勿循环轮询。每次用户请求只调用一次工具。\n"
        "- 始终显示完整的 task_id——不得截断。"
    ),
    subagents=async_subagents,
)


# ── REPL ──────────────────────────────────────────────────────────────────────

async def chat(user_input: str) -> None:
    """向监管智能体发送消息并打印响应。"""
    result = await supervisor.ainvoke(
        {"messages": [HumanMessage(user_input)]},
        config={"configurable": {"thread_id": thread_id}},
    )
    last = result["messages"][-1]
    content = last.content
    print(
        "\n"
        + (content if isinstance(content, str) else __import__("json").dumps(content, indent=2))
        + "\n"
    )


async def main() -> None:
    """运行交互式 REPL。"""
    print(f"监管智能体已连接到研究服务：{RESEARCHER_URL}")
    print("输入消息后按回车。Ctrl+C 或 Ctrl+D 退出。\n")
    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break
        if not user_input:
            continue
        try:
            await chat(user_input)
        except Exception as exc:  # noqa: BLE001
            print(f"错误：{exc}")


if __name__ == "__main__":
    asyncio.run(main())
