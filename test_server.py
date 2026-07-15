"""异步子智能体服务器的最小化端到端测试。

测试 Agent Protocol 的 HTTP 契约，无需调用真实大模型。
智能体的 ainvoke 被补丁替换为固定响应。
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

import server


@pytest.fixture(autouse=True)
def _fresh_db():
    """每次测试前重新初始化内存数据库。"""
    server._conn.executescript("DROP TABLE IF EXISTS runs; DROP TABLE IF EXISTS threads;")
    server._init_db()


FAKE_RESPONSE = {"messages": [AIMessage(content="以下是研究结果。")]}


def _make_ainvoke_mock():
    mock = AsyncMock(return_value=FAKE_RESPONSE)
    return mock


@pytest.fixture()
def client():
    return TestClient(server.app)


def test_health(client):
    resp = client.get("/ok")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_create_thread(client):
    resp = client.post("/threads")
    assert resp.status_code == 200
    data = resp.json()
    assert "thread_id" in data
    assert data["messages"] == []


def test_create_run_starts_agent(client):
    thread = client.post("/threads").json()
    thread_id = thread["thread_id"]

    with patch.object(server, "_agent") as mock_agent:
        mock_agent.ainvoke = _make_ainvoke_mock()
        resp = client.post(
            f"/threads/{thread_id}/runs",
            json={
                "assistant_id": "researcher",
                "input": {"messages": [{"role": "user", "content": "test query"}]},
            },
        )

    assert resp.status_code == 200
    run = resp.json()
    assert run["thread_id"] == thread_id
    assert "run_id" in run
    assert run["status"] == "pending"


def test_full_lifecycle(client):
    """创建会话 → 创建运行 → 等待完成 → 检查状态 → 获取会话。"""
    thread = client.post("/threads").json()
    thread_id = thread["thread_id"]

    with patch.object(server, "_agent") as mock_agent:
        mock_agent.ainvoke = _make_ainvoke_mock()
        run = client.post(
            f"/threads/{thread_id}/runs",
            json={
                "assistant_id": "researcher",
                "input": {"messages": [{"role": "user", "content": "quantum computing"}]},
            },
        ).json()
        run_id = run["run_id"]

        # 等待后台任务完成。
        asyncio.get_event_loop().run_until_complete(asyncio.sleep(0.5))

    # 检查运行状态——应为 success。
    status_resp = client.get(f"/threads/{thread_id}/runs/{run_id}")
    assert status_resp.status_code == 200
    assert status_resp.json()["status"] == "success"

    # 获取会话——应包含助手的回复消息。
    thread_resp = client.get(f"/threads/{thread_id}")
    assert thread_resp.status_code == 200
    thread_data = thread_resp.json()
    values_messages = thread_data["values"]["messages"]
    assert any(m["content"] == "以下是研究结果。" for m in values_messages)


def test_cancel_run(client):
    thread = client.post("/threads").json()
    thread_id = thread["thread_id"]

    # 创建一个慢速智能体，以便我们可以取消它。
    async def slow_ainvoke(*args, **kwargs):
        await asyncio.sleep(10)
        return FAKE_RESPONSE

    with patch.object(server, "_agent") as mock_agent:
        mock_agent.ainvoke = AsyncMock(side_effect=slow_ainvoke)
        run = client.post(
            f"/threads/{thread_id}/runs",
            json={
                "assistant_id": "researcher",
                "input": {"messages": [{"role": "user", "content": "something"}]},
            },
        ).json()
        run_id = run["run_id"]

    cancel_resp = client.post(f"/threads/{thread_id}/runs/{run_id}/cancel")
    assert cancel_resp.status_code == 200
    assert cancel_resp.json()["status"] == "cancelled"

    # 验证运行已取消。
    status_resp = client.get(f"/threads/{thread_id}/runs/{run_id}")
    assert status_resp.json()["status"] == "cancelled"


def test_interrupt_strategy(client):
    """使用 multitask_strategy='interrupt' 创建运行会取消正在运行的运行。"""
    thread = client.post("/threads").json()
    thread_id = thread["thread_id"]

    async def slow_ainvoke(*args, **kwargs):
        await asyncio.sleep(10)
        return FAKE_RESPONSE

    with patch.object(server, "_agent") as mock_agent:
        mock_agent.ainvoke = AsyncMock(side_effect=slow_ainvoke)
        first_run = client.post(
            f"/threads/{thread_id}/runs",
            json={
                "assistant_id": "researcher",
                "input": {"messages": [{"role": "user", "content": "first task"}]},
            },
        ).json()

        # 让第一次运行启动。
        asyncio.get_event_loop().run_until_complete(asyncio.sleep(0.1))

    with patch.object(server, "_agent") as mock_agent:
        mock_agent.ainvoke = _make_ainvoke_mock()
        second_run = client.post(
            f"/threads/{thread_id}/runs",
            json={
                "assistant_id": "researcher",
                "input": {"messages": [{"role": "user", "content": "new task"}]},
                "multitask_strategy": "interrupt",
            },
        ).json()

    # 第一次运行应被取消。
    first_status = client.get(f"/threads/{thread_id}/runs/{first_run['run_id']}").json()
    assert first_status["status"] == "cancelled"


def test_404_for_missing_thread(client):
    resp = client.get("/threads/nonexistent")
    assert resp.status_code == 404


def test_404_for_missing_run(client):
    thread = client.post("/threads").json()
    resp = client.get(f"/threads/{thread['thread_id']}/runs/nonexistent")
    assert resp.status_code == 404
