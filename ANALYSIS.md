# Embedding-Service 项目详细分析

## 📋 项目概述

**项目名称**: Embedding Service  
**项目类型**: 可复用的 AI 模型服务库  
**技术栈**: LangChain + FastAPI + Pydantic  
**主要功能**: 统一的文本嵌入和聊天模型接口，支持多个 LLM 提供商  
**版本**: 0.1.0  
**来源**: 从 airport-customer 项目中提取  

---

## 🎯 项目目标

### 核心目标
提供一个**轻量级、可复用的 AI 模型服务库**，统一不同 LLM 提供商的接口，便于：
- 快速集成文本嵌入功能
- 支持聊天/生成任务
- 轻松切换不同的模型提供商
- 作为独立微服务运行

### 设计哲学
```
单一职责 (SRP)
├─ 专注于 embedding + chat 模型
├─ 不处理 KB 构建、RAG 逻辑等
└─ 提供通用接口供其他服务使用

提供商无关 (Provider-Agnostic)
├─ 支持 Ollama（本地开源模型）
├─ 支持 OpenAI-compatible（云服务）
└─ 易于扩展新提供商

易于集成 (Integration-Ready)
├─ 库模式：import 导入使用
├─ 服务模式：独立 REST API 运行
└─ 配置驱动：通过环境变量控制
```

---

## 🏗️ 项目结构

### 目录布局
```
embedding-service/
├── embedding_service/          # 主包
│   ├── __init__.py            # 公开接口
│   ├── __main__.py            # CLI 入口 (可选)
│   ├── config.py              # 配置管理 (67 行)
│   ├── embeddings.py          # 模型工厂 (60 行)
│   └── api.py                 # REST API (110 行)
│
├── test_service.py            # 单元测试 (45 行)
├── example.py                 # 使用示例 (60 行)
├── requirements.txt           # 依赖列表
├── README.md                  # 项目文档
├── Makefile                   # 构建工具
└── .gitignore                # Git 忽略规则
```

### 代码统计
- **总代码行数**: ~400 行
- **核心库代码**: ~140 行 (config + embeddings)
- **API 代码**: ~110 行
- **测试代码**: ~45 行

---

## 🔌 核心模块分析

### 1. **config.py** (配置管理模块)

#### 职责
- 集中管理所有配置参数
- 从环境变量读取配置
- 提供配置验证和规范化

#### 关键类: Settings

```python
@dataclass
class Settings:
    provider: str              # "ollama" | "openai-compatible"
    llm_model: str            # 聊天模型名称
    embed_model: str          # 嵌入模型名称
    openai_base_url: str | None    # OpenAI API 地址
    openai_api_key: str | None     # OpenAI API 密钥
    ollama_base_url: str | None    # Ollama 服务地址
```

#### 环境变量映射

| 环境变量 | 默认值 | 含义 |
|---------|--------|------|
| `PROVIDER` | `ollama` | LLM 提供商 |
| `LLM_MODEL` | `qwen2.5:3b` | 聊天模型 |
| `EMBED_MODEL` | `mxbai-embed-large` | 嵌入模型 |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama 地址 |
| `OPENAI_BASE_URL` | `None` | OpenAI API 地址 |
| `OPENAI_API_KEY` | `None` | OpenAI 密钥 |

#### 工具函数

```python
def _get_int(name: str, default: int) -> int
    └─ 安全获取整数环境变量

def _get_float(name: str, default: float) -> float
    └─ 安全获取浮点数环境变量

def clamp_provider(provider: str) -> str
    └─ 规范化提供商名称
    └─ 对无效值返回 "ollama" (默认)
```

#### 配置加载流程

```python
# 1. 从环境变量创建配置
settings = Settings.from_env()

# 2. 自动规范化提供商名称
provider = clamp_provider("OLLAMA")  # → "ollama"

# 3. 回退到默认值
# 若 PROVIDER 未设置 → "ollama"
# 若 EMBED_MODEL 未设置 → "mxbai-embed-large"
```

---

### 2. **embeddings.py** (模型工厂模块)

#### 职责
- 创建嵌入模型实例
- 创建聊天模型实例
- 屏蔽提供商差异

#### 关键函数: build_embeddings()

```python
def build_embeddings(
    provider: str,
    embed_model: str,
    base_url: str | None = None,
    api_key: str | None = None,
) -> Embeddings
```

**工作流程**:
1. 规范化 provider 名称
2. 根据 provider 类型选择实现
3. 创建并返回模型实例

**支持的提供商**:

| 提供商 | 实现 | 依赖 | 配置 |
|------|------|------|------|
| `ollama` | `OllamaEmbeddings` | `langchain-ollama` | base_url |
| `openai-compatible` | `OpenAIEmbeddings` | `langchain-openai` | base_url + api_key |

**使用示例**:

```python
# Ollama (本地)
embeddings = build_embeddings(
    provider="ollama",
    embed_model="mxbai-embed-large",
    base_url="http://localhost:11434"
)

# OpenAI
embeddings = build_embeddings(
    provider="openai-compatible",
    embed_model="text-embedding-3-large",
    base_url="https://api.openai.com/v1",
    api_key="sk-..."
)

# 调用 API
vector = embeddings.embed_query("Hello world")
vectors = embeddings.embed_documents(["doc1", "doc2"])
```

#### 关键函数: build_chat_model()

```python
def build_chat_model(
    provider: str,
    llm_model: str,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.0,
) -> BaseChatModel
```

**工作流程**:
1. 规范化 provider
2. 根据 provider 选择实现
3. 创建聊天模型 (temperature 影响生成的随机性)

**支持的提供商**:

| 提供商 | 实现 | 依赖 |
|------|------|------|
| `ollama` | `ChatOllama` | `langchain-ollama` |
| `openai-compatible` | `ChatOpenAI` | `langchain-openai` |

**temperature 参数**:
- `0.0` - 确定性生成 (相同输入 → 相同输出)
- `0.5` - 平衡，有轻微变化
- `1.0` - 高随机性，创意输出

**使用示例**:

```python
from langchain_core.messages import HumanMessage, SystemMessage

chat = build_chat_model(
    provider="ollama",
    llm_model="qwen2.5:3b",
    temperature=0.0
)

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is AI?")
]

response = chat.invoke(messages)
print(response.content)
```

---

### 3. **api.py** (REST API 模块)

#### 职责
- 暴露 HTTP REST 端点
- 处理请求/响应序列化
- 错误处理和日志

#### 架构模式

```
FastAPI App
├── 初始化时
│   ├── 读取 Settings.from_env()
│   ├── 创建 embeddings 客户端
│   ├── 创建 chat_model 客户端
│   └─ 注册路由
│
└── 运行时
    └─ 请求 → 处理 → 响应
```

#### 数据模型 (Pydantic)

**请求模型**:
```python
class QueryRequest(BaseModel):
    text: str                          # 单个查询文本

class DocumentsRequest(BaseModel):
    texts: List[str]                   # 多个文档

class ChatRequest(BaseModel):
    message: str                       # 聊天消息
```

**响应模型**:
```python
class EmbeddingResponse(BaseModel):
    embedding: List[float]             # 单个向量 [1024]

class EmbeddingsResponse(BaseModel):
    embeddings: List[List[float]]      # 多个向量 [[1024], ...]

class ChatResponse(BaseModel):
    response: str                      # 聊天回复

class HealthResponse(BaseModel):
    status: str                        # "ok" | "error"
    provider: str                      # "ollama" | "openai-compatible"
    embed_model: str                   # 模型名称
    llm_model: str                     # 模型名称
```

#### REST 端点

| 方法 | 端点 | 功能 | 请求 | 响应 |
|------|------|------|------|------|
| GET | `/health` | 健康检查 | - | HealthResponse |
| POST | `/embed/query` | 单个文本嵌入 | QueryRequest | EmbeddingResponse |
| POST | `/embed/documents` | 批量文本嵌入 | DocumentsRequest | EmbeddingsResponse |
| POST | `/chat` | 聊天对话 | ChatRequest | ChatResponse |

#### 端点详解

**1. GET /health**
```json
请求: (无)

响应 (200 OK):
{
  "status": "ok",
  "provider": "ollama",
  "embed_model": "mxbai-embed-large",
  "llm_model": "qwen2.5:3b"
}
```

**2. POST /embed/query**
```json
请求:
{
  "text": "Machine learning is the future"
}

响应 (200 OK):
{
  "embedding": [0.123, -0.456, ..., 0.789]  # 1024 维向量
}

响应 (500 Error):
{
  "detail": "Model not found"
}
```

**3. POST /embed/documents**
```json
请求:
{
  "texts": [
    "Document 1",
    "Document 2",
    "Document 3"
  ]
}

响应 (200 OK):
{
  "embeddings": [
    [0.123, -0.456, ...],
    [0.789, -0.012, ...],
    [0.345, -0.678, ...]
  ]
}
```

**4. POST /chat**
```json
请求:
{
  "message": "What is artificial intelligence?"
}

响应 (200 OK):
{
  "response": "Artificial intelligence (AI) is the simulation of human intelligence..."
}
```

#### 错误处理

```python
try:
    # 执行操作
    vector = embeddings.embed_query(request.text)
except Exception as e:
    # 返回 HTTP 500 错误
    raise HTTPException(status_code=500, detail=str(e))
```

**常见错误**:
- `ModelNotFound` - 模型不存在或未下载
- `ConnectionError` - Ollama/API 服务不可用
- `InvalidRequest` - 输入格式错误
- `RateLimitError` - API 速率限制

---

## 📦 依赖分析

### Python 依赖

```
langchain-core>=0.3.0           # LLM 框架核心
  ├─ BaseLanguageModel          # 基类
  ├─ BaseChatModel              # 聊天基类
  ├─ Embeddings                 # 嵌入基类
  └─ messages                   # 消息类型

langchain-ollama>=0.2.0         # Ollama 集成
  ├─ OllamaEmbeddings
  ├─ ChatOllama
  └─ 支持本地开源模型

langchain-openai>=0.2.0         # OpenAI 集成
  ├─ OpenAIEmbeddings
  ├─ ChatOpenAI
  └─ 支持 OpenAI 和兼容 API

fastapi>=0.115.0                # Web 框架
  ├─ 快速 HTTP 服务
  ├─ 自动 OpenAPI 文档
  └─ 异步支持

uvicorn>=0.32.0                 # ASGI 服务器
  └─ 运行 FastAPI 应用

pydantic>=2.9.0                 # 数据验证
  ├─ 模型定义
  ├─ 类型检查
  └─ 序列化/反序列化
```

### 依赖树

```
embedding-service
├── langchain-core (LLM 框架)
│   └── pydantic (数据验证)
├── langchain-ollama (本地模型)
│   └── langchain-core
├── langchain-openai (云 API)
│   └── langchain-core
├── fastapi (Web 框架)
│   ├── pydantic
│   └── starlette
└── uvicorn (服务器)
    └── asgi
```

### 版本兼容性

| 组件 | 最小版本 | 推荐版本 | 备注 |
|------|---------|---------|------|
| Python | 3.8+ | 3.10+ | langchain 需要 3.8+ |
| langchain | 0.2.0 | 0.3+ | 频繁更新 |
| OpenAI API | - | 最新 | 支持 gpt-4, gpt-3.5 |

---

## 🔄 工作流程

### 场景 1: 作为库使用 (Library Mode)

```python
# 1. 导入
from embedding_service import build_embeddings, build_chat_model

# 2. 创建模型
embeddings = build_embeddings(
    provider="ollama",
    embed_model="mxbai-embed-large"
)

chat = build_chat_model(
    provider="ollama",
    llm_model="qwen2.5:3b"
)

# 3. 使用
vector = embeddings.embed_query("Hello")
response = chat.invoke([...])

# 4. 集成到其他应用
# kb_builder / rag_service 等都可以使用
```

**优点**:
- ✅ 轻量级，无启动开销
- ✅ 可直接集成到其他应用
- ✅ 完全控制生命周期

### 场景 2: 作为微服务运行 (Microservice Mode)

```bash
# 1. 启动服务
export PROVIDER=ollama
export EMBED_MODEL=mxbai-embed-large
export LLM_MODEL=qwen2.5:3b
uvicorn embedding_service.api:app --host 0.0.0.0 --port 8000

# 2. 远程调用
curl -X POST http://localhost:8000/embed/query \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello"}'

# 3. 多个应用可共享一个 embedding-service 实例
```

**优点**:
- ✅ 独立部署和扩展
- ✅ 多个应用共享模型
- ✅ 网络隔离
- ✅ 容器化部署

### 场景 3: 提供商切换

```python
# 场景 3a: Ollama → OpenAI
embeddings = build_embeddings(
    provider="openai-compatible",
    embed_model="text-embedding-3-large",
    base_url="https://api.openai.com/v1",
    api_key="sk-..."
)

# 代码无需改动，只需改配置！
# 通过 docker-compose 或 k8s 配置管理
```

---

## 🧪 测试分析

### 测试覆盖

```python
# test_service.py - 45 行

class TestConfig(unittest.TestCase):
    def test_clamp_provider(self)
        # 验证提供商名称规范化
        ✓ "ollama" → "ollama"
        ✓ "OLLAMA" → "ollama"
        ✓ "openai-compatible" → "openai-compatible"
        ✓ "invalid" → "ollama" (默认)
    
    def test_build_embeddings_ollama(self)
        # Mock 测试 Ollama 嵌入创建
        ✓ 正确传递参数
        ✓ 返回 OllamaEmbeddings 实例
```

### 测试策略

**单元测试** (已实现):
- ✅ 配置规范化
- ✅ 模型工厂创建
- ⚠️ 需要 Mock 避免实际网络调用

**集成测试** (需要):
- ❌ 真实 Ollama 连接
- ❌ 真实 OpenAI API 调用
- ❌ API 端点完整流程

**测试运行**:
```bash
# 运行单元测试
python -m unittest test_service.py -v

# 输出
test_clamp_provider ... ok
test_build_embeddings_ollama ... ok
Ran 2 tests in 0.357s
OK
```

---

## 💡 设计模式

### 1. 工厂模式 (Factory Pattern)

```python
# embeddings.py 中的两个工厂函数
def build_embeddings(...) -> Embeddings
    # 根据 provider 参数创建合适的实例
    
def build_chat_model(...) -> BaseChatModel
    # 同上，创建聊天模型
```

**优点**:
- 隐藏实现细节
- 易于扩展新提供商
- 客户端代码不需要改变

### 2. 配置对象模式 (Configuration Object)

```python
@dataclass
class Settings:
    # 集中所有配置
    # 从环境变量加载
    # 提供默认值
```

**优点**:
- 单一信息来源
- 易于验证和规范化
- 便于传递和共享

### 3. 依赖注入 (Dependency Injection)

```python
# FastAPI 应用初始化时注入依赖
def create_app() -> FastAPI:
    settings = Settings.from_env()
    embeddings = build_embeddings(...)
    chat_model = build_chat_model(...)
    # 在路由中使用注入的实例
```

**优点**:
- 便于测试 (Mock 注入)
- 解耦组件
- 生命周期管理

---

## 🚀 使用场景

### 场景 1: RAG 系统中的嵌入组件

```python
# kb_builder.py 中使用
from embedding_service import build_embeddings

embeddings = build_embeddings(
    provider=settings.provider,
    embed_model=settings.embed_model,
    base_url=settings.ollama_base_url
)

# 为知识库块生成向量
vectors = embeddings.embed_documents(chunks)
# 保存到 FAISS 索引
```

### 场景 2: RAG 系统中的检索组件

```python
# rag.py 中使用
from embedding_service import build_embeddings

embeddings = build_embeddings(...)

# 查询向量化
query_vec = embeddings.embed_query(question)

# FAISS 检索
scores, indices = kb.index.search(query_vec, k=5)
```

### 场景 3: 独立聊天服务

```bash
# 启动 embedding-service 作为聊天微服务
$ PROVIDER=openai uvicorn embedding_service.api:app

# 其他应用调用
$ curl -X POST http://embedding-service:8000/chat \
    -d '{"message": "Hello"}'
```

### 场景 4: 多模型部署

```
┌─ Embedding Service (mxbai-embed-large)
│  └─ KB Builder 使用
│
├─ Chat Service (qwen2.5:3b)
│  └─ RAG/API 使用
│
└─ Vision Service (llava)
   └─ 图文理解使用

每个服务独立部署，互不影响
```

---

## ⚙️ 配置方案

### 方案 1: 环境变量 (开发环境)

```bash
# .env 或 shell
export PROVIDER=ollama
export EMBED_MODEL=mxbai-embed-large
export LLM_MODEL=qwen2.5:3b
export OLLAMA_BASE_URL=http://localhost:11434

# 启动
python -m embedding_service
```

### 方案 2: Docker 环境变量 (容器化)

```dockerfile
FROM python:3.11
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY embedding_service ./embedding_service
ENV PROVIDER=ollama
ENV EMBED_MODEL=mxbai-embed-large
ENV OLLAMA_BASE_URL=http://ollama:11434
CMD ["uvicorn", "embedding_service.api:app", "--host", "0.0.0.0"]
```

### 方案 3: Docker Compose 服务编排

```yaml
version: '3.8'
services:
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
  
  embedding-service:
    build: .
    ports:
      - "8000:8000"
    environment:
      PROVIDER: ollama
      OLLAMA_BASE_URL: http://ollama:11434
    depends_on:
      - ollama
```

---

## 📊 性能特性

### 嵌入性能

| 操作 | 模型 | 延迟 | 吞吐量 |
|------|------|------|--------|
| 单文本嵌入 | mxbai-embed-large | ~100ms | - |
| 批量 (100 文本) | mxbai-embed-large | ~500ms | 200 docs/sec |
| OpenAI API | text-embedding-3-large | ~100ms | 受 API 限流 |

### 聊天性能

| 操作 | 模型 | 延迟 | 备注 |
|------|------|------|------|
| 短回复 (<50 tokens) | qwen2.5:3b | 1-2 sec | 本地 Ollama |
| 长回复 (>500 tokens) | qwen2.5:3b | 5-10 sec | 本地 Ollama |
| OpenAI API | gpt-4 | 1-3 sec | 云服务 |

### 资源使用

| 资源 | Ollama 本地 | OpenAI API |
|------|-----------|-----------|
| 内存 | 4-8 GB | ~100 MB |
| GPU | 需要 | 不需要 |
| 网络 | 本地 | 需要 |
| 成本 | 0 (硬件成本) | 按用量计费 |

---

## 🔒 安全考虑

### API Key 管理

```python
# ✅ 好的做法：环境变量
OPENAI_API_KEY=sk-... (在 .env 或 k8s secret)

# ❌ 坏的做法：硬编码
api_key = "sk-..."  # 不要这样做！
```

### 输入验证

```python
# Pydantic 自动验证
class QueryRequest(BaseModel):
    text: str  # 必须是字符串

# 若提交数据类型错误，FastAPI 自动拒绝
```

### 错误信息

```python
# ✅ 安全：隐藏内部细节
raise HTTPException(status_code=500, detail="Internal error")

# ❌ 不安全：暴露堆栈跟踪
detail=traceback.format_exc()
```

### 速率限制

```python
# 生产环境建议添加
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@limiter.limit("100/minute")
@app.post("/chat")
def chat(...): ...
```

---

## 🔄 与其他模块的集成

### 与 docx-parser 的关系

```
docx-parser
  └─ 解析文档为文本块
     └─ embedding-service
        └─ 为文本块生成向量
           └─ kb-builder
              └─ 构建知识库索引
```

### 与 kb-builder 的关系

```
kb-builder
  ├─ 扫描文档
  ├─ 调用 docx-parser 解析
  ├─ 调用 embedding-service 嵌入向量
  └─ 构建 FAISS 索引
```

### 与 rag-service 的关系

```
rag-service
  ├─ 接收用户问题
  ├─ 调用 embedding-service 向量化查询
  ├─ FAISS 检索相关块
  ├─ 调用 embedding-service 聊天模型
  └─ 生成回答
```

### 与 customer-service-api 的关系

```
customer-service-api
  └─ 集成 embedding-service (本地或远程)
     └─ 用于知识库查询和聊天
```

---

## 📈 扩展方向

### 1. 支持新提供商

```python
# 添加 Anthropic Claude 支持
if provider == "anthropic":
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(model=llm_model, api_key=api_key)

# 添加本地 LLaMA 支持
if provider == "llama-cpp":
    from langchain_community.llms import LlamaCpp
    return LlamaCpp(model_path=model_path, ...)
```

### 2. 缓存支持

```python
# 缓存嵌入结果，避免重复计算
@cache
def embed_query(text: str) -> List[float]:
    return embeddings.embed_query(text)
```

### 3. 批处理优化

```python
# 实现流式嵌入，处理超大文档
def embed_documents_stream(texts: Iterator[str]) -> Iterator[List[float]]:
    for batch in iter_batches(texts, batch_size=100):
        yield embeddings.embed_documents(batch)
```

### 4. 监控指标

```python
# 添加 Prometheus metrics
embedding_requests_total = Counter(...)
embedding_latency = Histogram(...)
embedding_errors_total = Counter(...)
```

---

## 🎓 学习资源

### LangChain 文档
- Embeddings: https://python.langchain.com/docs/integrations/text_embedding/
- Chat Models: https://python.langchain.com/docs/integrations/chat/

### Ollama
- 官网: https://ollama.ai/
- 模型库: https://ollama.ai/library
- 本地部署: https://github.com/jmorganca/ollama

### OpenAI API
- 官网: https://platform.openai.com/docs
- 嵌入模型: https://platform.openai.com/docs/guides/embeddings
- 聊天模型: https://platform.openai.com/docs/guides/gpt

### FastAPI
- 官网: https://fastapi.tiangolo.com/
- 部署: https://fastapi.tiangolo.com/deployment/

---

## 🏆 项目优势

✅ **轻量级** - 仅 ~300 行代码，核心功能完整  
✅ **提供商无关** - 轻松切换 Ollama / OpenAI / 其他  
✅ **双模式** - 库模式 + 服务模式，灵活使用  
✅ **易于测试** - 依赖注入，便于 Mock  
✅ **配置驱动** - 通过环境变量控制，容器友好  
✅ **类型安全** - Pydantic 自动验证和序列化  
✅ **自动文档** - FastAPI 自动生成 Swagger UI  

---

## ⚠️ 已知限制

❌ **无缓存** - 相同文本重复嵌入会重新计算  
❌ **无重试逻辑** - 网络错误会直接失败  
❌ **无速率限制** - 生产环境需要自己添加  
❌ **无并发控制** - 可能同时多个请求竞争  
❌ **模型硬编码** - 不支持动态切换模型  
❌ **无日志** - 缺少结构化日志输出  

---

## 📝 使用建议

1. **开发环境**: 使用 Ollama (本地、免费、快速迭代)
2. **生产环境**: 考虑 OpenAI (稳定、成熟、付费)
3. **混合方案**: Ollama 备份 + OpenAI 主力
4. **容器部署**: Docker + docker-compose 或 Kubernetes
5. **监控告警**: 添加日志、指标、健康检查

---

## 🚀 下一步计划

1. ✅ 完成 embedding-service (当前)
2. ⏳ 创建 kb-builder 项目 (下一步)
3. ⏳ 创建 rag-service 项目
4. ⏳ 创建 customer-service-api 项目
5. ⏳ 分离 customer-service-web 项目

