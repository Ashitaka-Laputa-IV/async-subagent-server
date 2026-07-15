"""异步子智能体服务器（Async Subagent Server）——基于 Agent Protocol 的 FastAPI 服务。

一个最小化的自托管 Agent Protocol 服务器，将 Deep Agents 的研究智能体
暴露为异步子智能体。任意 Deep Agents 监管智能体均可以通过
AsyncSubAgent 配置连接到该服务器。

实现了 Deep Agents 异步子智能体中间件（通过 LangGraph SDK）调用的端点：

    POST /threads                              创建会话
    POST /threads/{thread_id}/runs             启动（或中断+重启）运行
    GET  /threads/{thread_id}/runs/{run_id}    轮询运行状态
    GET  /threads/{thread_id}                  获取会话（成功时读取 values.messages）
    POST /threads/{thread_id}/runs/{run_id}/cancel  取消运行
    GET  /ok                                   健康检查

持久化使用内存 SQLite 数据库（无需文件，无需配置）。
启动时自动创建数据库模式。

运行：
    uv run uvicorn server:app --port 2024

然后让 Deep Agents 监管智能体指向：
    RESEARCHER_URL=http://localhost:2024
"""

from __future__ import annotations

import asyncio
import sqlite3
import uuid
import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

load_dotenv(Path(__file__).parent / ".env")

# ── 数据库 ────────────────────────────────────────────────────────────────────

# 进程内共享的内存 SQLite，所有连接共用。
_conn = sqlite3.connect(":memory:", check_same_thread=False)
_conn.row_factory = sqlite3.Row


def _init_db() -> None:
    """创建 threads 和 runs 表（如不存在）。

    threads — 每个对话会话一行
        messages  JSON 数组，元素为 {role, content} 对象
        values    JSON 对象，存储会话最终状态（values.messages）

    runs    — 每个会话的每次运行尝试一行
        status    取值：pending | running | success | error | cancelled
    """
    _conn.executescript("""
        CREATE TABLE IF NOT EXISTS threads (
            thread_id  TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            messages   TEXT NOT NULL DEFAULT '[]',
            values_    TEXT NOT NULL DEFAULT '{}'
        );
        CREATE TABLE IF NOT EXISTS runs (
            run_id       TEXT PRIMARY KEY,
            thread_id    TEXT NOT NULL REFERENCES threads(thread_id),
            assistant_id TEXT NOT NULL,
            status       TEXT NOT NULL DEFAULT 'pending',
            created_at   TEXT NOT NULL,
            error        TEXT
        );
    """)
    _conn.commit()


# ── 数据库辅助函数 ───────────────────────────────────────────────────────────

import json  # noqa: E402  （在标准库后、第三方库前导入）


def _get_thread(thread_id: str) -> dict[str, Any] | None:
    row = _conn.execute(
        "SELECT thread_id, created_at, messages, values_ FROM threads WHERE thread_id = ?",
        (thread_id,),
    ).fetchone()
    if row is None:
        return None
    return {
        "thread_id": row["thread_id"],
        "created_at": row["created_at"],
        "messages": json.loads(row["messages"]),
        "values": json.loads(row["values_"]),
    }


def _get_run(run_id: str) -> dict[str, Any] | None:
    row = _conn.execute(
        "SELECT run_id, thread_id, assistant_id, status, created_at, error FROM runs WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    if row is None:
        return None
    return dict(row)


# ── 智能体 ────────────────────────────────────────────────────────────────────
#
# 将此处替换为你自己的智能体。唯一要求是：接受一个 messages 数组，
# 返回一个包含 messages 数组的对象。

import os  # noqa: E402


@tool
async def web_search(query: str) -> str:
    """搜索网页获取信息。用于查找当前数据、新闻和分析。

    Args:
        query: 搜索关键词。
    """
    if os.environ.get("TAVILY_API_KEY"):
        import httpx

        async with httpx.AsyncClient() as client:
            res = await client.post(
                "https://api.tavily.com/search",
                json={"api_key": os.environ["TAVILY_API_KEY"], "query": query, "max_results": 5},
                timeout=30,
            )
        data = res.json()
        results = data.get("results") or []
        if not results:
            return f'未找到 "{query}" 的相关结果'
        return "\n\n".join(
            f"{i + 1}. **{r['title']}**\n   {r['content']}\n   来源：{r['url']}"
            for i, r in enumerate(results)
        )


from deepagents import create_deep_agent  # noqa: E402

_agent = create_deep_agent(
    model=ChatOpenAI(
        model=os.getenv("LLM_MODEL_NAME"),
        base_url=os.getenv("LLM_BASE_URL"),
        api_key=os.getenv("LLM_API_KEY"),
    ),
    system_prompt=(
        "你是一名细致的研究智能体。使用网络搜索调查话题，"
        "并写出一份结构清晰的研究摘要（300–500 字）。尽量注明信息来源。\n\n"
        "如果在对话中途收到新的指令，立即执行，不要追问——"
        "丢弃之前的工作，从头开始处理新任务。"
    ),
    tools=[web_search],
)


# ── 运行执行器 ───────────────────────────────────────────────────────────────

async def _execute_run(run_id: str, thread_id: str, user_message: str) -> None:
    """调用智能体并持久化结果；以 fire-and-forget 方式执行。"""
    _conn.execute("UPDATE runs SET status = 'running' WHERE run_id = ?", (run_id,))
    _conn.commit()
    try:
        result = await _agent.ainvoke({"messages": [HumanMessage(user_message)]})
        last = result["messages"][-1]
        output = last.content if isinstance(last.content, str) else json.dumps(last.content)
        assistant_msg = {"role": "assistant", "content": output}
        # 获取当前消息，追加助手回复，然后持久化。
        # values.messages 是 LangGraph SDK 在成功时读取的字段。
        row = _conn.execute(
            "SELECT messages FROM threads WHERE thread_id = ?", (thread_id,)
        ).fetchone()
        msgs = json.loads(row['messages']) if row else []
        msgs.append(assistant_msg)
        serialized = json.dumps(msgs)
        _conn.execute(
            "UPDATE threads SET messages = ?, values_ = ? WHERE thread_id = ?",
            (serialized, json.dumps({"messages": msgs}), thread_id),
        )
        _conn.execute("UPDATE runs SET status = 'success' WHERE run_id = ?", (run_id,))
        _conn.commit()
    except Exception as exc:  # noqa: BLE001
        _conn.execute(
            "UPDATE runs SET status = 'error', error = ? WHERE run_id = ?",
            (str(exc), run_id),
        )
        _conn.commit()


# ── 应用 ──────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def _lifespan(app: FastAPI):  # type: ignore[type-arg]
    _init_db()
    if not os.environ.get("TAVILY_API_KEY"):
        raise RuntimeError(
            "启动失败：未设置 TAVILY_API_KEY。请设置该环境变量后再启动服务。"
        )
    yield


app = FastAPI(lifespan=_lifespan)


# ── 路由 ──────────────────────────────────────────────────────────────────────

@app.get("/ok")
async def health() -> dict[str, bool]:
    """健康检查。"""
    return {"ok": True}


@app.post("/threads")
async def create_thread() -> dict[str, Any]:
    """创建会话。start_async_task 在创建运行前调用此接口。"""
    thread_id = str(uuid.uuid4())
    now = datetime.now(UTC).isoformat()
    _conn.execute(
        "INSERT INTO threads (thread_id, created_at) VALUES (?, ?)",
        (thread_id, now),
    )
    _conn.commit()
    return {"thread_id": thread_id, "created_at": now, "messages": [], "values": {}}


@app.post("/threads/{thread_id}/runs")
async def create_run(thread_id: str, request: Request) -> dict[str, Any]:
    """在已有会话上创建一次运行。

    由 start_async_task（新任务）和 update_async_task
    （用新指令重新运行）调用。当 multitask_strategy 为 'interrupt' 时，
    该会话上正在运行的任何运行都会被取消，会话状态也会在
    新运行启动前清空。
    """
    thread = _get_thread(thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="未找到会话")

    body = await request.json()
    multitask_strategy = body.get("multitask_strategy")

    if multitask_strategy == "interrupt":
        _conn.execute(
            "UPDATE runs SET status = 'cancelled' WHERE thread_id = ? AND status = 'running'",
            (thread_id,),
        )
        _conn.execute(
            "UPDATE threads SET values_ = '{}' WHERE thread_id = ?",
            (thread_id,),
        )
        _conn.commit()

    messages = (body.get("input") or {}).get("messages") or []
    user_message = next((m["content"] for m in messages if m.get("role") == "user"), "")

    if user_message:
        existing = json.loads(
            _conn.execute(
                "SELECT messages FROM threads WHERE thread_id = ?", (thread_id,)
            ).fetchone()['messages']
        )
        existing.append({"role": "user", "content": user_message})
        _conn.execute(
            "UPDATE threads SET messages = ? WHERE thread_id = ?",
            (json.dumps(existing), thread_id),
        )
        _conn.commit()

    run_id = str(uuid.uuid4())
    now = datetime.now(UTC).isoformat()
    assistant_id = body.get("assistant_id") or "researcher"
    _conn.execute(
        "INSERT INTO runs (run_id, thread_id, assistant_id, created_at) VALUES (?, ?, ?, ?)",
        (run_id, thread_id, assistant_id, now),
    )
    _conn.commit()

    # Fire and forget——客户端通过 GET /threads/{thread_id}/runs/{run_id} 轮询状态。
    asyncio.ensure_future(_execute_run(run_id, thread_id, user_message))

    return {
        "run_id": run_id,
        "thread_id": thread_id,
        "assistant_id": assistant_id,
        "status": "pending",
        "created_at": now,
        "error": None,
    }


@app.get("/threads/{thread_id}/runs/{run_id}")
async def get_run(thread_id: str, run_id: str) -> dict[str, Any]:
    """获取运行状态。check_async_task 通过此接口轮询任务是否完成。"""
    run = _get_run(run_id)
    if run is None or run["thread_id"] != thread_id:
        raise HTTPException(status_code=404, detail="未找到运行")
    return run


@app.get("/threads/{thread_id}")
async def get_thread(thread_id: str) -> dict[str, Any]:
    """获取会话状态。check_async_task 在运行状态变为 'success' 后调用此接口。

    SDK 读取 values['messages'] 来提取最终结果。
    """
    thread = _get_thread(thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="未找到会话")
    return thread


@app.post("/threads/{thread_id}/runs/{run_id}/cancel")
async def cancel_run(thread_id: str, run_id: str) -> dict[str, Any]:
    """取消运行。cancel_async_task 调用此接口。

    在数据库中将运行标记为已取消。注意：智能体的调用不会在半路中断——
    要实现真正的取消，需接入 asyncio.Task 的取消机制。
    """
    run = _get_run(run_id)
    if run is None or run["thread_id"] != thread_id:
        raise HTTPException(status_code=404, detail="未找到运行")
    _conn.execute("UPDATE runs SET status = 'cancelled' WHERE run_id = ?", (run_id,))
    _conn.commit()
    return {**run, "status": "cancelled"}
