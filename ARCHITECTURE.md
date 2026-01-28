# Embedding-Service 架构分析

## 1️⃣ 整体架构概览

### 系统级架构 (System-Level Architecture)

```
┌─────────────────────────────────────────────────────────────┐
│                    客户端应用                                 │
│  (KB Builder / RAG Service / Customer API / 其他服务)        │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │ (库模式)       │ (服务模式)      │
        │ import         │ HTTP/REST      │
        │                │                │
        ▼                ▼                │
┌──────────────────────────────────┐    │
│     embedding-service 库          │    │
│  ┌────────────────────────────┐  │    │
│  │  FastAPI REST API          │──┼────┘
│  │  (api.py)                  │  │
│  │  ┌─ GET /health           │  │
│  │  ├─ POST /embed/query     │  │
│  │  ├─ POST /embed/documents │  │
│  │  └─ POST /chat            │  │
│  └────────────────────────────┘  │
│  ┌────────────────────────────┐  │
│  │  配置管理                    │  │
│  │  (config.py)               │  │
│  │  ├─ Settings 数据类        │  │
│  │  ├─ from_env() 加载配置    │  │
│  │  └─ clamp_provider() 规范化│  │
│  └────────────────────────────┘  │
│  ┌────────────────────────────┐  │
│  │  模型工厂                    │  │
│  │  (embeddings.py)           │  │
│  │  ├─ build_embeddings()     │  │
│  │  └─ build_chat_model()     │  │
│  └────────────────────────────┘  │
└──────────────────────────────────┘
         │              │
         ▼              ▼
    ┌────────────┐  ┌──────────────┐
    │  Ollama    │  │  OpenAI API  │
    │  本地模型   │  │  云服务模型   │
    └────────────┘  └──────────────┘
```

---

## 2️⃣ 分层架构 (Layered Architecture)

```
┌──────────────────────────────────────────┐
│         API 层 (Presentation Layer)      │
│  - HTTP 请求处理                         │
│  - 数据验证 (Pydantic)                   │
│  - 响应序列化                            │
│  - 错误处理                              │
└──────┬───────────────────────────────────┘
       │
┌──────▼───────────────────────────────────┐
│       应用层 (Application Layer)          │
│  - 业务逻辑: 嵌入 + 聊天                  │
│  - 工厂函数: 模型创建                     │
│  - 生命周期管理                          │
└──────┬───────────────────────────────────┘
       │
┌──────▼───────────────────────────────────┐
│    基础设施层 (Infrastructure Layer)      │
│  - 配置管理                              │
│  - 环境变量读取                          │
│  - 依赖初始化                            │
└──────┬───────────────────────────────────┘
       │
┌──────▼───────────────────────────────────┐
│       外部服务层 (External Services)      │
│  - LangChain 框架                        │
│  - Ollama 本地服务                       │
│  - OpenAI 云服务                         │
└──────────────────────────────────────────┘
```

---

## 3️⃣ 组件架构 (Component Architecture)

### 模块间依赖关系

```
embedding_service/
├── __init__.py
│   └─ 公开接口
│      ├─ build_embeddings (来自 embeddings.py)
│      ├─ build_chat_model (来自 embeddings.py)
│      ├─ Settings (来自 config.py)
│      └─ clamp_provider (来自 config.py)
│
├── config.py (67 行)
│   ├─ Settings (数据类)
│   ├─ _get_int() (工具)
│   ├─ _get_float() (工具)
│   └─ clamp_provider() (工具)
│      用途: 配置管理和环境变量读取
│
├── embeddings.py (60 行)
│   ├─ build_embeddings()
│   │  ├─ 依赖: config.clamp_provider
│   │  ├─ 依赖: langchain_ollama.OllamaEmbeddings
│   │  ├─ 依赖: langchain_openai.OpenAIEmbeddings
│   │  └─ 返回: Embeddings 实例
│   │
│   └─ build_chat_model()
│      ├─ 依赖: config.clamp_provider
│      ├─ 依赖: langchain_ollama.ChatOllama
│      ├─ 依赖: langchain_openai.ChatOpenAI
│      └─ 返回: BaseChatModel 实例
│      用途: 模型工厂
│
└── api.py (110 行)
   ├─ 数据模型 (Pydantic)
   │  ├─ QueryRequest
   │  ├─ DocumentsRequest
   │  ├─ ChatRequest
   │  ├─ EmbeddingResponse
   │  ├─ EmbeddingsResponse
   │  ├─ ChatResponse
   │  └─ HealthResponse
   │
   ├─ create_app()
   │  ├─ 依赖: Settings.from_env()
   │  ├─ 依赖: build_embeddings()
   │  ├─ 依赖: build_chat_model()
   │  └─ 返回: FastAPI 应用
   │
   └─ 路由处理
      ├─ GET /health
      ├─ POST /embed/query
      ├─ POST /embed/documents
      └─ POST /chat
      用途: HTTP API 端点
```

### 依赖图 (Dependency Graph)

```
                    __init__.py
                   /    |    \
                  /     |     \
             config  embeddings  api
               |          |       |
               |          |    FastAPI
               |          |       |
               |   +──────┼───────┘
               |   |      |
               |   |   langchain
               |   |      |
        os.getenv   |   LLMProvider
                    |   (Ollama/OpenAI)
                    |
              clamp_provider
```

---

## 4️⃣ 数据流分析 (Data Flow Analysis)

### 数据流 #1: 嵌入查询

```
用户 (客户端)
  │
  ├─ 库模式:
  │    from embedding_service import build_embeddings
  │    embeddings = build_embeddings(...)
  │    vector = embeddings.embed_query("text")
  │
  └─ 服务模式:
       POST /embed/query
       │
       ▼
    FastAPI 解析请求
       │
       ├─ body: QueryRequest { text: str }
       │
       ▼
    build_embeddings 的路由处理器
       │
       ├─ 提取 request.text
       │
       ▼
    embeddings.embed_query(text)
       │
       ├─ LangChain 框架
       │
       ▼
    Ollama / OpenAI API
       │
       ├─ 文本向量化
       │
       ▼
    返回向量 [float, float, ...]
       │
       ├─ EmbeddingResponse { embedding: [...] }
       │
       ▼
    JSON 序列化
       │
       ▼
    HTTP 200 响应
       │
       ▼
    用户接收答案
```

### 数据流 #2: 批量嵌入

```
用户请求:
  POST /embed/documents
  {
    "texts": ["doc1", "doc2", "doc3"]
  }
    │
    ▼
FastAPI 请求解析
    │
    ├─ DocumentsRequest validation
    │
    ▼
embeddings.embed_documents(texts)
    │
    ├─ 批处理 (LangChain 内部)
    │  └─ 可能分成多个批次
    │
    ▼
Ollama / OpenAI API 调用
    │
    ├─ 返回 List[List[float]]
    │
    ▼
EmbeddingsResponse 封装
    │
    ├─ embeddings: [[...], [...], [...]]
    │
    ▼
HTTP 200 响应
```

### 数据流 #3: 聊天

```
用户消息:
  "What is AI?"
    │
    ▼
POST /chat
{
  "message": "What is AI?"
}
    │
    ▼
FastAPI 解析
    │
    ├─ ChatRequest validation
    │
    ▼
chat_model.invoke([...])
    │
    ├─ 构建 messages 列表
    │  └─ HumanMessage(content="...")
    │
    ▼
LangChain ChatModel
    │
    ├─ 与 Ollama / OpenAI 通信
    │
    ▼
生成回复
    │
    ├─ 流式或完整返回
    │
    ▼
ChatResponse { response: "..." }
    │
    ▼
HTTP 200 响应
```

---

## 5️⃣ 请求生命周期 (Request Lifecycle)

### 库模式 (Library Mode)

```
┌─────────────────────────────────────┐
│ 1. 导入模块                         │
│    from embedding_service import .. │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│ 2. 创建模型工厂                      │
│    embeddings = build_embeddings()   │
│    chat = build_chat_model()         │
└──────────────┬──────────────────────┘
               │ (阻塞直到模型初始化完成)
┌──────────────▼──────────────────────┐
│ 3. 使用模型                         │
│    vector = embeddings.embed_query() │
│    response = chat.invoke()         │
└──────────────┬──────────────────────┘
               │ (返回结果)
┌──────────────▼──────────────────────┐
│ 4. 处理结果                         │
│    保存向量 / 显示回复               │
└─────────────────────────────────────┘
```

**生命周期特点**:
- ✅ 同步阻塞
- ✅ 完全控制
- ✅ 无网络开销
- ❌ 启动时间长 (模型初始化)

### 服务模式 (Microservice Mode)

```
启动阶段:
┌────────────────────────────────────┐
│ uvicorn embedding_service.api:app  │
└──────────────┬─────────────────────┘
               │
┌──────────────▼─────────────────────┐
│ 1. 初始化 FastAPI 应用              │
│    app = create_app()              │
└──────────────┬─────────────────────┘
               │
┌──────────────▼─────────────────────┐
│ 2. 从环境变量加载配置                │
│    settings = Settings.from_env()   │
└──────────────┬─────────────────────┘
               │
┌──────────────▼─────────────────────┐
│ 3. 初始化模型实例                    │
│    embeddings = build_embeddings()  │
│    chat_model = build_chat_model()  │
└──────────────┬─────────────────────┘
               │
┌──────────────▼─────────────────────┐
│ 4. 注册路由                         │
│    @app.get("/health")             │
│    @app.post("/embed/query")       │
│    @app.post("/embed/documents")   │
│    @app.post("/chat")              │
└──────────────┬─────────────────────┘
               │
┌──────────────▼─────────────────────┐
│ 5. 启动服务器                       │
│    监听 0.0.0.0:8000               │
└────────────────────────────────────┘

运行时:
┌────────────────────────────────────┐
│ 客户端 HTTP 请求                   │
└──────────────┬─────────────────────┘
               │
┌──────────────▼─────────────────────┐
│ 1. FastAPI 接收请求                 │
│    路由匹配 + 参数提取              │
└──────────────┬─────────────────────┘
               │
┌──────────────▼─────────────────────┐
│ 2. Pydantic 验证请求数据            │
│    类型检查 + 格式验证              │
└──────────────┬─────────────────────┘
               │
┌──────────────▼─────────────────────┐
│ 3. 执行路由处理函数                 │
│    调用模型 API                    │
└──────────────┬─────────────────────┘
               │
┌──────────────▼─────────────────────┐
│ 4. 序列化响应                       │
│    Pydantic 模型 → JSON            │
└──────────────┬─────────────────────┘
               │
┌──────────────▼─────────────────────┐
│ 5. 返回 HTTP 响应                   │
│    200 OK + JSON 体                │
└────────────────────────────────────┘
```

---

## 6️⃣ 配置管理架构

### 配置层次 (Configuration Hierarchy)

```
优先级最高
   │
   ▼
┌──────────────────────────┐
│ 代码中显式传入的参数      │
│ (在函数调用时覆盖)       │
└──────────────┬───────────┘
               │
┌──────────────▼───────────┐
│ 环境变量                 │
│ (os.getenv)             │
│ PROVIDER=ollama         │
│ EMBED_MODEL=...         │
└──────────────┬───────────┘
               │
┌──────────────▼───────────┐
│ 代码默认值               │
│ (config.py 中定义)      │
│ PROVIDER="ollama"       │
│ EMBED_MODEL="mxbai..." │
└──────────────┬───────────┘
               │
优先级最低

示例:
Settings.from_env()
  └─ PROVIDER=None
     └─ 返回默认值 "ollama"
```

### 配置数据流

```
操作系统环境变量
  │
  ├─ PROVIDER
  ├─ LLM_MODEL
  ├─ EMBED_MODEL
  ├─ OLLAMA_BASE_URL
  ├─ OPENAI_BASE_URL
  └─ OPENAI_API_KEY
  │
  ▼
config._get_int()
config._get_float()
  │
  ├─ 类型转换和错误处理
  │
  ▼
Settings 数据类
  │
  ├─ provider: str
  ├─ llm_model: str
  ├─ embed_model: str
  ├─ openai_base_url: Optional[str]
  ├─ openai_api_key: Optional[str]
  └─ ollama_base_url: Optional[str]
  │
  ▼
FastAPI 应用 (api.py)
  │
  ├─ 创建 embeddings 客户端
  ├─ 创建 chat_model 客户端
  │
  ▼
业务逻辑使用
```

---

## 7️⃣ 模型初始化流程

### 初始化序列图 (Initialization Sequence)

```
客户端代码
   │
   ├─ build_embeddings(
   │    provider="ollama",
   │    embed_model="mxbai-embed-large"
   │  )
   │
   ▼
config.clamp_provider("ollama")
   │
   ├─ 验证 provider 值
   ├─ 返回 "ollama"
   │
   ▼
条件判断: provider == "ollama"
   │
   ├─ 是 → 执行下一步
   ├─ 否 → 检查下一个条件
   │
   ▼
from langchain_ollama import OllamaEmbeddings
   │
   ├─ 导入 LangChain Ollama 模块
   │
   ▼
OllamaEmbeddings(
   model="mxbai-embed-large",
   base_url="http://localhost:11434"
 )
   │
   ├─ 创建客户端
   ├─ 验证连接 (可选)
   │
   ▼
返回 Embeddings 实例
   │
   ├─ embeddings.embed_query(text) 准备就绪
   ├─ embeddings.embed_documents(texts) 准备就绪
   │
   ▼
客户端获得可用的嵌入对象
```

### 错误处理路径 (Error Handling Path)

```
模型初始化
   │
   ├─ 成功
   │  └─ 返回模型实例
   │
   └─ 失败
      │
      ├─ Ollama 服务不可用
      │  └─ ConnectionError
      │
      ├─ OpenAI API 无效
      │  └─ AuthenticationError
      │
      ├─ 模型不存在
      │  └─ ModelNotFoundError
      │
      └─ 其他异常
         └─ 捕获 → 日志 → 抛出 RuntimeError
```

---

## 8️⃣ 提供商抽象架构

### 提供商接口一致性

```
统一的 Embeddings 接口 (LangChain)
  │
  ├─ embed_query(text: str) → List[float]
  └─ embed_documents(texts: List[str]) → List[List[float]]
  │
  ▼
多个实现:

┌──────────────────────┐      ┌──────────────────────┐
│ OllamaEmbeddings     │      │ OpenAIEmbeddings     │
│ (langchain-ollama)   │      │ (langchain-openai)   │
│                      │      │                      │
│ 实现:                │      │ 实现:                │
│ - embed_query()      │      │ - embed_query()      │
│ - embed_documents()  │      │ - embed_documents()  │
│                      │      │                      │
│ 后端:                │      │ 后端:                │
│ HTTP → Ollama        │      │ HTTPS → OpenAI       │
│ (本地服务)            │      │ (云服务)              │
└──────────────────────┘      └──────────────────────┘
         ▲                               ▲
         │                               │
         └───────────┬──────────────────┘
                     │
                     │ 透明切换
                     │ (通过 provider 参数)
                     │
                build_embeddings()
```

### 提供商切换示例

```
# 情景 1: 使用 Ollama
embeddings_ollama = build_embeddings(
    provider="ollama",
    embed_model="mxbai-embed-large"
)
vec = embeddings_ollama.embed_query("hello")

# 情景 2: 切换到 OpenAI (代码无需改动)
embeddings_openai = build_embeddings(
    provider="openai-compatible",
    embed_model="text-embedding-3-large",
    base_url="https://api.openai.com/v1",
    api_key="sk-..."
)
vec = embeddings_openai.embed_query("hello")

# 两个调用返回相同格式的向量!
# 这就是多态的好处
```

---

## 9️⃣ 通信架构

### 库模式通信 (In-Process)

```
应用进程
┌─────────────────────────────────┐
│ embedding-service (库)           │
│ ┌───────────────────────────┐   │
│ │ build_embeddings()        │   │
│ │ build_chat_model()        │   │
│ └───────────────────────────┘   │
└──────┬────────────────────┬─────┘
       │                    │
       ▼                    ▼
    Ollama                OpenAI
    本地 TCP               HTTPS
```

**特点**:
- ✅ 零网络开销 (函数调用)
- ✅ 低延迟
- ✅ 简单调试
- ❌ 需要在同一进程

### 服务模式通信 (Inter-Process / Remote)

```
客户端应用                  embedding-service 应用
┌──────────────┐           ┌──────────────────────┐
│ 业务逻辑      │           │ FastAPI 应用         │
└──┬───────────┘           └──────┬───────────────┘
   │                              │
   │ HTTP 请求                     │
   ├─ GET /health                 │
   ├─ POST /embed/query           │
   ├─ POST /embed/documents       │
   └─ POST /chat                  │
   │                              │
   ├──────────────────────────────►
   │     Content-Type: application/json
   │
   │                    解析请求
   │                    验证 Pydantic 模型
   │                    调用模型 API
   │                    序列化响应
   │
   │◄──────────────────────────────
   │
   │ HTTP 响应 (200 OK)
   │ Content-Type: application/json
   │
```

**特点**:
- ✅ 独立进程，独立生命周期
- ✅ 多客户端共享一个服务
- ✅ 容器化部署
- ❌ 网络开销
- ❌ 延迟增加 (HTTP 往返)

---

## 🔟 并发模型 (Concurrency Model)

### 库模式下的并发

```
应用线程
  │
  ├─ Thread 1: embeddings.embed_query("A")
  │  └─ 调用 Ollama (阻塞)
  │     └─ 等待响应
  │
  ├─ Thread 2: embeddings.embed_query("B")
  │  └─ 调用 Ollama (阻塞)
  │     └─ 等待响应
  │
  └─ Thread 3: embeddings.embed_query("C")
     └─ 调用 Ollama (阻塞)
        └─ 等待响应

特点: 阻塞 I/O，需要多线程处理并发
```

### 服务模式下的并发

```
FastAPI (ASGI 框架)
  │
  ├─ Uvicorn Worker 1
  │  └─ 处理请求 1: POST /embed/query
  │     └─ embeddings.embed_query()
  │
  ├─ Uvicorn Worker 2
  │  └─ 处理请求 2: POST /embed/query
  │     └─ embeddings.embed_query()
  │
  └─ Uvicorn Worker 3
     └─ 处理请求 3: POST /chat
        └─ chat_model.invoke()

特点: 
- 多 Worker 处理并发
- 可在 Uvicorn 配置中调整 worker 数量
- 生产环境推荐 workers = 2 * cpu_count + 1
```

---

## 1️⃣1️⃣ 单一职责设计 (SRP)

### 模块职责划分

```
config.py
├─ 职责: 管理配置
├─ 关键函数: Settings.from_env()
├─ 不做: 创建模型、处理 HTTP 请求
└─ 不做: 执行 LLM 操作

embeddings.py
├─ 职责: 创建模型工厂
├─ 关键函数: build_embeddings(), build_chat_model()
├─ 不做: 管理配置 (依赖 config.py)
├─ 不做: 处理 HTTP (依赖 api.py)
└─ 不做: 直接调用 LLM

api.py
├─ 职责: HTTP 端点和数据验证
├─ 关键函数: create_app()
├─ 不做: 管理配置 (依赖 config.py)
├─ 不做: 创建模型 (依赖 embeddings.py)
└─ 不做: 业务逻辑

各模块各司其职，易于测试和维护
```

---

## 1️⃣2️⃣ 扩展性架构

### 添加新提供商的步骤

```
1. 在 embeddings.py 中添加新分支:

def build_embeddings(...):
    provider = clamp_provider(provider)
    
    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(...)
    
    if provider == "openai-compatible":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(...)
    
    # 新增：Anthropic 提供商
    if provider == "anthropic":
        from langchain_anthropic import AnthropicEmbeddings
        return AnthropicEmbeddings(...)
    
    raise ValueError(f"Unsupported provider: {provider}")

2. 无需修改 config.py 或 api.py
3. 客户端代码完全不变
4. 通过环境变量切换: PROVIDER=anthropic
```

### 添加新 API 端点的步骤

```
1. 在 api.py 中添加新的 Pydantic 模型:

class EmbeddingSimilarityRequest(BaseModel):
    query_vector: List[float]
    document_vectors: List[List[float]]

class EmbeddingSimilarityResponse(BaseModel):
    similarities: List[float]

2. 在 create_app() 中添加新路由:

@app.post("/embed/similarity")
def embed_similarity(request: EmbeddingSimilarityRequest):
    similarities = cosine_similarity(
        request.query_vector,
        request.document_vectors
    )
    return EmbeddingSimilarityResponse(
        similarities=similarities
    )

3. 无需修改其他模块
4. 新功能立即可用
```

---

## 1️⃣3️⃣ 部署架构

### 开发环境 (单机)

```
┌─────────────────────────────────┐
│ 本地开发机                       │
│                                 │
│ ┌─────────────────────────────┐ │
│ │ embedding-service 库         │ │
│ │ + 测试代码                   │ │
│ │ + 示例代码                   │ │
│ └─────────────────────────────┘ │
│                                 │
│ Ollama (本地)                    │
│ http://localhost:11434          │
│                                 │
└─────────────────────────────────┘
```

### 测试环境 (Docker)

```
┌─────────────────────────────────┐
│ Docker Container 1 (Ollama)     │
│ - ollama/ollama                 │
│ - Port 11434                    │
└─────────────────────────────────┘
         ▲
         │ TCP
         │
┌─────────────────────────────────┐
│ Docker Container 2 (API)        │
│ - embedding-service API         │
│ - FastAPI + Uvicorn             │
│ - Port 8000                     │
└─────────────────────────────────┘
         ▲
         │ HTTP
         │
┌─────────────────────────────────┐
│ Docker Container 3 (KB Builder) │
│ - 调用 embedding-service:8000  │
└─────────────────────────────────┘
```

### 生产环境 (Kubernetes)

```
┌─────────────────────────────────────────────┐
│         Kubernetes Cluster                  │
│                                             │
│ ┌───────────────┐  ┌──────────────────┐   │
│ │ embedding-service │  │ embedding-service │   │
│ │ Pod (副本 1)   │  │ Pod (副本 2)    │   │
│ │ :8000         │  │ :8000            │   │
│ └───────────────┘  └──────────────────┘   │
│        ▲                    ▲              │
│        │                    │              │
│  ┌─────┴────────────────────┴────┐        │
│  │   Service (负载均衡)          │        │
│  │   embedding-service:8000      │        │
│  └──────────────┬────────────────┘        │
│                 │                         │
│ ┌───────────────┴──────────────┐          │
│ │  ConfigMap (配置)            │          │
│ │  - PROVIDER=ollama           │          │
│ │  - EMBED_MODEL=...           │          │
│ └────────────────────────────┘           │
│                                          │
│ ┌────────────────────────────┐           │
│ │  StatefulSet (Ollama)      │           │
│ │  - PVC (持久化存储)         │           │
│ │  - Port 11434              │           │
│ └────────────────────────────┘           │
│                                          │
└─────────────────────────────────────────┘
         ▲
         │ HTTP (Ingress)
         │
    ┌────┴────────┐
    │  客户端      │
    └─────────────┘
```

---

## 1️⃣4️⃣ 故障处理架构

### 故障场景和应对

```
场景 1: Ollama 服务崩溃
原因: Ollama 进程死亡
检测: build_embeddings() 初始化失败
应对:
  - 重启 Ollama
  - 使用备用提供商 (OpenAI)
  - 自动故障转移

场景 2: OpenAI API 限流
原因: 请求过多
检测: 429 Too Many Requests
应对:
  - 实现退避重试 (exponential backoff)
  - 使用本地 Ollama 备份
  - 队列限流

场景 3: 网络超时
原因: API 响应慢
检测: ConnectTimeout / ReadTimeout
应对:
  - 设置合理的超时时间
  - 重试机制
  - 降级处理

场景 4: 内存不足
原因: 大规模嵌入操作
检测: MemoryError
应对:
  - 分批处理
  - 流式处理
  - 使用更小的模型
```

---

## 1️⃣5️⃣ 安全架构

### 多层安全控制

```
┌──────────────────────────────────────┐
│ 1. API 入口层 (API Gateway)         │
│    - IP 白名单                       │
│    - TLS/SSL 加密                    │
│    - 请求签名验证                    │
└────────┬─────────────────────────────┘
         │
┌────────▼─────────────────────────────┐
│ 2. 认证层 (Authentication)           │
│    - API Key 验证                    │
│    - OAuth 2.0                       │
│    - JWT Token                       │
└────────┬─────────────────────────────┘
         │
┌────────▼─────────────────────────────┐
│ 3. 授权层 (Authorization)            │
│    - 基于角色的访问控制 (RBAC)       │
│    - 速率限制                        │
│    - 配额管理                        │
└────────┬─────────────────────────────┘
         │
┌────────▼─────────────────────────────┐
│ 4. 输入验证 (Input Validation)      │
│    - Pydantic 类型检查              │
│    - 长度限制                        │
│    - 格式验证                        │
└────────┬─────────────────────────────┘
         │
┌────────▼─────────────────────────────┐
│ 5. API 密钥管理 (Secret Management)  │
│    - 环境变量                        │
│    - K8s Secrets                     │
│    - Vault                           │
└──────────────────────────────────────┘
```

---

## 1️⃣6️⃣ 监控和可观测性架构

### 三大支柱 (Three Pillars)

```
┌─────────────────────────────────────┐
│  1. 日志 (Logging)                  │
│     - 请求日志                      │
│     - 错误日志                      │
│     - 审计日志                      │
│     → ELK Stack 或 Loki             │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│  2. 指标 (Metrics)                  │
│     - 请求数 / 响应时间             │
│     - 错误率                        │
│     - 资源使用率                    │
│     → Prometheus + Grafana          │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│  3. 追踪 (Tracing)                  │
│     - 请求链路                      │
│     - 服务间调用                    │
│     - 性能瓶颈                      │
│     → Jaeger 或 Zipkin              │
└─────────────────────────────────────┘
```

### 关键指标 (Key Metrics)

```
API 层:
  - request_count (总请求数)
  - request_latency (延迟分布)
  - error_rate (错误率)
  - http_status_codes (状态码分布)

模型层:
  - embedding_latency (嵌入延迟)
  - embedding_requests (嵌入请求数)
  - chat_latency (聊天延迟)
  - model_errors (模型错误)

系统层:
  - cpu_usage (CPU 使用率)
  - memory_usage (内存使用率)
  - disk_io (磁盘 I/O)
  - network_io (网络 I/O)

业务层:
  - unique_users (独立用户)
  - tokens_generated (生成的 token 数)
  - cost (成本，如使用 OpenAI)
```

---

## 1️⃣7️⃣ 与上下游系统的集成

### 整体微服务架构

```
┌─────────────────────────────────────────────────────┐
│                    客户端应用                       │
└───┬─────────────────────────────────────────────┬──┘
    │                                             │
    ▼                                             ▼
┌─────────────────────────┐            ┌──────────────────────┐
│  KB Builder Service     │            │  RAG Service         │
│  - 文档处理             │            │  - 检索               │
│  - 分块                 │            │  - 生成               │
│  - 索引构建             │            │  - 回答优化           │
└──────────┬──────────────┘            └──────────┬───────────┘
           │ (调用)                              │ (调用)
           │                                     │
           └─────────────┬───────────────────────┘
                         │
                         ▼
            ┌────────────────────────────┐
            │ 🎯 Embedding Service       │
            │ (你在这里)                 │
            │                            │
            │ - 嵌入向量化               │
            │ - 聊天补全                 │
            │ - 提供商无关               │
            └────────────┬───────────────┘
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
    ┌──────────┐  ┌──────────┐  ┌──────────┐
    │  Ollama  │  │ OpenAI   │  │ Claude   │
    │  (本地)   │  │ (云)      │  │ (云)     │
    └──────────┘  └──────────┘  └──────────┘
```

---

## 1️⃣8️⃣ 架构质量属性 (Quality Attributes)

### 可靠性 (Reliability)

```
- MTBF (Mean Time Between Failures): 目标 > 99.9%
- MTTR (Mean Time To Recover): 目标 < 5 分钟
- 健康检查: GET /health 每 10 秒
- 自动故障转移: Ollama → OpenAI
```

### 性能 (Performance)

```
- 嵌入延迟: < 200ms (100 个文本)
- 聊天延迟: < 3 秒 (qwen2.5:3b)
- 吞吐量: > 100 req/sec
- P95 延迟: < 500ms
```

### 可扩展性 (Scalability)

```
- 垂直扩展: 更好的硬件 / 更大的模型
- 水平扩展: 多个 embedding-service 实例
- 负载均衡: 轮询 / 最少连接
- 批处理优化: 向量操作矢量化
```

### 可维护性 (Maintainability)

```
- 代码行数: < 500 行 (易于理解)
- 圈复杂度: < 10 (代码质量)
- 单元测试覆盖率: > 80%
- 文档完整性: README + ANALYSIS + 代码注释
```

### 可观测性 (Observability)

```
- 日志级别: INFO / WARN / ERROR
- 结构化日志: JSON 格式
- 追踪 ID: 请求级追踪
- 健康指标: /health 端点
```

---

## 总结

### 架构优势

✅ **简洁** - 少量代码，高清晰度  
✅ **灵活** - 支持多提供商，易于扩展  
✅ **可复用** - 库模式 + 服务模式  
✅ **易测** - 依赖注入，便于 Mock  
✅ **容器友好** - 环境变量配置，Docker ready  

### 架构短板

❌ **无缓存** - 相同输入重复计算  
❌ **无重试** - 网络错误直接失败  
❌ **无限流** - 生产环境需添加  
❌ **无日志** - 缺少可观测性  
❌ **无监控** - 没有指标收集  

### 下一步改进

1. 添加日志和结构化日志
2. 实现重试和熔断机制
3. 添加缓存层 (Redis)
4. 添加 Prometheus 指标
5. 实现速率限制和配额管理

