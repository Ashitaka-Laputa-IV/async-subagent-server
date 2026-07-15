# 异步子智能体服务器（Async Subagent Server）

一个自托管的 [Agent Protocol（智能体协议）](https://github.com/langchain-ai/agent-protocol)服务器，将 Deep Agents 的研究智能体暴露为异步子智能体。以此为起点，你可以在任意基础设施上托管自己的智能体，并将其连接到 Deep Agents 的监管智能体。

本示例展示了模式的两端：

- **`server.py`**——你的子智能体运行的 FastAPI 服务器
- **`supervisor.py`**——一个交互式 REPL，展示如何连接到该服务器

## 前置条件

- `LLM_API_KEY`——必填
- `TAVILY_API_KEY`——可选；未设置时会使用桩搜索（stub search）

## 虚拟环境配置（Python 3.12.7）

本项目使用 [uv](https://docs.astral.sh/uv/) 管理 Python 版本和虚拟环境。

- [ ] 安装 Python 3.12.7：`uv python install 3.12.7`
- [ ] 创建虚拟环境：`uv venv --python 3.12.7`
- [ ] 激活虚拟环境：`.venv\Scripts\activate`（Windows）或 `source .venv/bin/activate`（macOS/Linux）
- [ ] 安装依赖：`uv sync`

## 快速开始

**1. 配置环境变量：**

将 `.env.example` 复制为 `.env`，然后填入你的密钥：

```bash
# Windows (PowerShell)
copy .env.example .env

# macOS / Linux
# cp .env.example .env
```

**2. 启动服务器：**

```bash
uv run uvicorn server:app --port 2024
```

**3. 另开一个终端，启动监管智能体：**

```bash
uv run python supervisor.py
```

> 两个脚本均自动通过 `load_dotenv()` 加载 `.env` 文件，无需在命令行中传递环境变量。

试试这些提示词：

```
> 研究量子计算的最新发展
> 查看 <task-id> 的状态
> 更新 <task-id>，使其只关注商业应用
> 取消 <task-id>
> 列出所有任务
```

## 已实现的端点

以下是 Deep Agents 异步子智能体中间件（通过 LangGraph SDK）调用的 Agent Protocol 端点：

| 端点 | 用途 |
| -------------------------------------------- | -------------------------------- |
| `POST /threads` | 为新任务创建一个会话（Thread） |
| `POST /threads/{thread_id}/runs` | 启动或中断+重启一次运行 |
| `GET /threads/{thread_id}/runs/{run_id}` | 轮询运行状态 |
| `GET /threads/{thread_id}` | 获取会话状态（`values.messages`） |
| `POST /threads/{thread_id}/runs/{run_id}/cancel` | 取消运行 |
| `GET /ok` | 健康检查 |

## 替换为你自己的智能体

将 `server.py` 中的 `create_deep_agent` 调用替换成你自己的智能体。Agent Protocol 层不会随智能体行为变化：

```python
_agent = create_deep_agent(
    model=ChatOpenAI(
        model=os.getenv("LLM_MODEL_NAME"),
        base_url=os.getenv("LLM_BASE_URL"),
        api_key=os.getenv("LLM_API_KEY"),
    ),
    system_prompt="你是一个……",
    tools=[your_tool],
)
```

## ⚠️ 仅供演示

本示例旨在展示自托管异步子智能体模式，不包含认证、限流等生产环境所需的特性。

## 资源

- [LangChain 学院](https://academy.langchain.com/)——由 LangChain 团队制作的 LangChain 库与产品免费课程
- [行为准则](https://github.com/langchain-ai/langchain/?tab=coc-ov-file)——社区准则与标准
