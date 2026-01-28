# api.py 详细分析

## 📋 文件概览

**文件名**: api.py  
**代码行数**: 110 行  
**职责**: 暴露 REST API 端点，处理 HTTP 请求/响应  
**核心类**: 8 个 Pydantic 模型 + 1 个工厂函数 + 4 个路由处理器  

---

## 🏗️ 代码结构

### 文件组织 (File Organization)

```
api.py
├─ 导入部分 (Lines 1-11)
│  ├─ 标准库
│  ├─ 第三方库 (fastapi, pydantic)
│  └─ 本地模块 (config, embeddings)
│
├─ 数据模型部分 (Lines 13-43)
│  ├─ 请求模型 (3 个)
│  │  ├─ QueryRequest
│  │  ├─ DocumentsRequest
│  │  └─ ChatRequest
│  ├─ 响应模型 (4 个)
│  │  ├─ EmbeddingResponse
│  │  ├─ EmbeddingsResponse
│  │  ├─ ChatResponse
│  │  └─ HealthResponse
│  └─ 辅助函数
│
├─ 应用工厂部分 (Lines 45-108)
│  ├─ create_app() 函数
│  ├─ 配置加载
│  ├─ 模型初始化
│  ├─ 错误处理
│  └─ 路由注册
│
└─ 应用实例部分 (Line 111)
   └─ app = create_app()
```

---

## 📦 Pydantic 数据模型分析

### 1. 请求模型 (Request Models)

#### QueryRequest (Line 15-16)
```python
class QueryRequest(BaseModel):
    text: str
```

**目的**: 封装单个文本查询请求  
**字段**:
- `text: str` - 待嵌入的文本（必需）

**使用场景**: POST /embed/query  
**验证规则**: FastAPI 自动验证
- 必须是字符串
- 不能为空（默认）
- 自动生成 OpenAPI schema

**示例**:
```json
{
  "text": "Machine learning is awesome"
}
```

---

#### DocumentsRequest (Line 19-20)
```python
class DocumentsRequest(BaseModel):
    texts: List[str]
```

**目的**: 封装多文本批量嵌入请求  
**字段**:
- `texts: List[str]` - 待嵌入的文本列表（必需）

**使用场景**: POST /embed/documents  
**验证规则**:
- 必须是字符串列表
- 每个字符串验证同 QueryRequest

**示例**:
```json
{
  "texts": [
    "Document 1",
    "Document 2",
    "Document 3"
  ]
}
```

**性能考虑**:
- 无限制的列表长度（应添加最大值）
- 批量大小由 LangChain 内部处理

---

#### ChatRequest (Line 23-24)
```python
class ChatRequest(BaseModel):
    message: str
```

**目的**: 封装聊天消息请求  
**字段**:
- `message: str` - 用户消息（必需）

**使用场景**: POST /chat  
**验证规则**: 同 QueryRequest

**示例**:
```json
{
  "message": "What is artificial intelligence?"
}
```

---

### 2. 响应模型 (Response Models)

#### EmbeddingResponse (Line 27-28)
```python
class EmbeddingResponse(BaseModel):
    embedding: List[float]
```

**目的**: 返回单个嵌入向量  
**字段**:
- `embedding: List[float]` - 向量数据（通常 1024 维）

**使用场景**: POST /embed/query 的响应  
**数据大小**: ~4KB (1024 floats × 4 bytes)

**示例**:
```json
{
  "embedding": [0.123, -0.456, ..., 0.789]  // 1024 个浮点数
}
```

---

#### EmbeddingsResponse (Line 31-32)
```python
class EmbeddingsResponse(BaseModel):
    embeddings: List[List[float]]
```

**目的**: 返回多个嵌入向量  
**字段**:
- `embeddings: List[List[float]]` - 向量列表

**使用场景**: POST /embed/documents 的响应  
**数据大小**: N × 4KB (N = 文本数)

**示例**:
```json
{
  "embeddings": [
    [0.123, -0.456, ..., 0.789],
    [0.234, -0.567, ..., 0.890],
    [0.345, -0.678, ..., 0.901]
  ]
}
```

---

#### ChatResponse (Line 35-36)
```python
class ChatResponse(BaseModel):
    response: str
```

**目的**: 返回聊天回复  
**字段**:
- `response: str` - 模型生成的文本

**使用场景**: POST /chat 的响应  
**大小**: 通常 100-2000 字符

**示例**:
```json
{
  "response": "Artificial intelligence is the simulation of human intelligence by computer systems..."
}
```

---

#### HealthResponse (Line 39-43)
```python
class HealthResponse(BaseModel):
    status: str
    provider: str
    embed_model: str
    llm_model: str
```

**目的**: 返回服务健康状态和配置信息  
**字段**:
- `status: str` - "ok" 或 "error"
- `provider: str` - "ollama" 或 "openai-compatible"
- `embed_model: str` - 嵌入模型名称
- `llm_model: str` - LLM 模型名称

**使用场景**: GET /health 的响应  
**用途**: 
- 负载均衡器健康检查
- 监控系统状态
- 验证配置

**示例**:
```json
{
  "status": "ok",
  "provider": "ollama",
  "embed_model": "mxbai-embed-large",
  "llm_model": "qwen2.5:3b"
}
```

---

## 🏭 工厂函数分析

### create_app() 函数 (Lines 46-110)

#### 函数签名
```python
def create_app() -> FastAPI:
    """Create FastAPI application."""
```

**返回类型**: FastAPI 应用实例  
**职责**:
1. 加载配置
2. 初始化模型
3. 注册路由
4. 返回完整的应用

#### 第一步: 配置加载 (Lines 47-48)

```python
settings = Settings.from_env()
app = FastAPI(title="Embedding Service API")
```

**作用**:
- 从环境变量读取配置
- 创建 FastAPI 应用实例

**配置示例**:
```python
# Settings 实例包含:
settings.provider              # "ollama" 或 "openai-compatible"
settings.embed_model           # "mxbai-embed-large"
settings.llm_model            # "qwen2.5:3b"
settings.ollama_base_url      # "http://localhost:11434"
settings.openai_base_url      # None 或 "https://api.openai.com/v1"
settings.openai_api_key       # None 或 "sk-..."
```

#### 第二步: 模型初始化 (Lines 50-60)

```python
try:
    embeddings = build_embeddings(
        settings.provider,
        settings.embed_model,
        base_url=settings.ollama_base_url if settings.provider == "ollama" else settings.openai_base_url,
        api_key=settings.openai_api_key,
    )
    chat_model = build_chat_model(
        settings.provider,
        settings.llm_model,
        base_url=settings.ollama_base_url if settings.provider == "ollama" else settings.openai_base_url,
        api_key=settings.openai_api_key,
    )
except Exception as e:
    raise RuntimeError(f"Failed to initialize models: {e}")
```

**关键逻辑**:

| 提供商 | base_url | api_key |
|--------|----------|---------|
| `ollama` | `ollama_base_url` | 不需要 |
| `openai-compatible` | `openai_base_url` | `openai_api_key` |

**错误处理**:
- ✅ 捕获模型初始化异常
- ✅ 包装为 RuntimeError
- ❌ 不提供详细的错误消息分类

**可能的异常**:
- `ConnectionError` - 服务不可用
- `AuthenticationError` - API 密钥无效
- `ModelNotFoundError` - 模型未找到

#### 第三步: 路由注册 (Lines 62-107)

4 个路由处理器被注册到应用。

---

## 🛣️ 路由处理器分析

### 1. GET /health 端点 (Lines 62-71)

```python
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        provider=settings.provider,
        embed_model=settings.embed_model,
        llm_model=settings.llm_model,
    )
```

**URL**: `GET /health`  
**响应模型**: `HealthResponse`  
**HTTP 状态码**: 200 (成功)

**用途**:
- 负载均衡器探针
- 监控系统检查
- 验证配置

**流程**:
```
GET /health
  ↓
health()
  ↓
读取 settings 属性
  ↓
构造 HealthResponse
  ↓
FastAPI 序列化为 JSON
  ↓
HTTP 200 OK
```

**响应示例**:
```json
HTTP/1.1 200 OK
Content-Type: application/json

{
  "status": "ok",
  "provider": "ollama",
  "embed_model": "mxbai-embed-large",
  "llm_model": "qwen2.5:3b"
}
```

**性能**: 极快 (< 1ms)  
**副作用**: 无  
**可靠性**: 很高（无外部依赖）

---

### 2. POST /embed/query 端点 (Lines 73-80)

```python
@app.post("/embed/query", response_model=EmbeddingResponse)
def embed_query(request: QueryRequest) -> EmbeddingResponse:
    """Embed a single query text."""
    try:
        vector = embeddings.embed_query(request.text)
        return EmbeddingResponse(embedding=vector)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

**URL**: `POST /embed/query`  
**请求体**: `QueryRequest { text: str }`  
**响应体**: `EmbeddingResponse { embedding: List[float] }`  

**流程**:
```
POST /embed/query
├─ Content-Type: application/json
├─ Body: { "text": "..." }
  ↓
FastAPI 解析请求
  ↓
Pydantic 验证 QueryRequest
  ├─ 检查 text 是字符串
  ├─ 检查 text 不为空（可选）
  └─ 绑定到 request 参数
  ↓
embed_query(request)
  ↓
embeddings.embed_query(request.text)
  ├─ LangChain 框架
  ├─ 调用 Ollama / OpenAI
  └─ 返回向量 [float, float, ...]
  ↓
异常处理:
├─ 成功: 返回 EmbeddingResponse
└─ 失败: 捕获 → HTTPException(500)
  ↓
FastAPI 序列化响应
  ↓
HTTP 响应 (200 或 500)
```

**请求示例**:
```bash
curl -X POST http://localhost:8000/embed/query \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}'
```

**成功响应** (200 OK):
```json
{
  "embedding": [0.123, -0.456, ..., 0.789]
}
```

**失败响应** (500 Internal Server Error):
```json
{
  "detail": "Model not found: mxbai-embed-large"
}
```

**性能**:
- 延迟: ~100-200ms
- 受限于 Ollama / OpenAI 响应时间

**错误场景**:
- ❌ Ollama 服务离线
- ❌ OpenAI API 配额用尽
- ❌ 网络超时
- ❌ 模型未下载

---

### 3. POST /embed/documents 端点 (Lines 82-89)

```python
@app.post("/embed/documents", response_model=EmbeddingsResponse)
def embed_documents(request: DocumentsRequest) -> EmbeddingsResponse:
    """Embed multiple documents."""
    try:
        vectors = embeddings.embed_documents(request.texts)
        return EmbeddingsResponse(embeddings=vectors)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

**URL**: `POST /embed/documents`  
**请求体**: `DocumentsRequest { texts: List[str] }`  
**响应体**: `EmbeddingsResponse { embeddings: List[List[float]] }`  

**流程**: 同 `/embed/query`，但批量处理

**请求示例**:
```bash
curl -X POST http://localhost:8000/embed/documents \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Document 1",
      "Document 2",
      "Document 3"
    ]
  }'
```

**成功响应** (200 OK):
```json
{
  "embeddings": [
    [0.123, -0.456, ..., 0.789],
    [0.234, -0.567, ..., 0.890],
    [0.345, -0.678, ..., 0.901]
  ]
}
```

**性能**:
- 延迟: ~500ms (100 文本) 到 ~2s (1000 文本)
- LangChain 内部进行批处理

**批处理优化**:
```
texts: [100 个文本]
  ↓
LangChain (内部)
  ├─ 分成 batch_size=20 的批次
  ├─ 第1批 → API 调用 1
  ├─ 第2批 → API 调用 2
  ├─ ...
  └─ 第5批 → API 调用 5
  ↓
合并所有向量
  ↓
返回 [List[float]] × 100
```

**性能考虑** ⚠️:
- 无最大长度限制（应添加 `max_items` 验证）
- 大批量请求可能导致超时

---

### 4. POST /chat 端点 (Lines 91-107)

```python
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Chat completion endpoint."""
    try:
        from langchain_core.messages import HumanMessage
        
        response = chat_model.invoke([HumanMessage(content=request.message)])
        return ChatResponse(response=response.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

**URL**: `POST /chat`  
**请求体**: `ChatRequest { message: str }`  
**响应体**: `ChatResponse { response: str }`  

**流程**:
```
POST /chat
├─ Content-Type: application/json
├─ Body: { "message": "..." }
  ↓
FastAPI 解析 + 验证
  ↓
chat(request)
  ↓
from langchain_core.messages import HumanMessage
  └─ 动态导入（性能影响）
  ↓
HumanMessage(content=request.message)
  ├─ 构造 LangChain 消息对象
  └─ 格式: {"type": "human", "content": "..."}
  ↓
chat_model.invoke([msg])
  ├─ 调用 Ollama / OpenAI
  ├─ 执行 LLM 推理
  └─ 返回 AIMessage
  ↓
response.content
  └─ 提取文本部分
  ↓
ChatResponse(response=...)
  ↓
序列化为 JSON
  ↓
HTTP 200 OK
```

**请求示例**:
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is machine learning?"}'
```

**成功响应** (200 OK):
```json
{
  "response": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience..."
}
```

**性能**:
- 延迟: 2-10 秒 (Ollama qwen3b)
- 延迟: 1-3 秒 (OpenAI gpt-4)

**动态导入问题** ⚠️:
```python
# 问题: 每次请求都导入
from langchain_core.messages import HumanMessage

# 改进: 在文件开头导入
# from langchain_core.messages import HumanMessage
```

---

## 🔒 错误处理分析

### 现有错误处理

```python
try:
    # 执行 LLM 操作
    vector = embeddings.embed_query(request.text)
except Exception as e:
    # 捕获所有异常
    raise HTTPException(status_code=500, detail=str(e))
```

**问题**:
1. ❌ 过于宽泛（捕获 Exception）
2. ❌ 无错误分类
3. ❌ 无日志记录
4. ❌ 无重试机制
5. ⚠️ 错误消息可能暴露内部细节

### 改进建议

```python
# 1. 具体异常处理
from langchain.errors import LLMError, APIConnectionError

try:
    vector = embeddings.embed_query(request.text)
except APIConnectionError as e:
    logger.error(f"Connection failed: {e}")
    raise HTTPException(status_code=503, detail="Service unavailable")
except LLMError as e:
    logger.error(f"LLM error: {e}")
    raise HTTPException(status_code=500, detail="Model error")
except Exception as e:
    logger.exception(f"Unexpected error: {e}")
    raise HTTPException(status_code=500, detail="Internal error")

# 2. 添加日志
import logging
logger = logging.getLogger(__name__)

# 3. 重试机制
from tenacity import retry, stop_after_attempt

@retry(stop=stop_after_attempt(3))
def embed_query_with_retry(text: str):
    return embeddings.embed_query(text)
```

---

## 📊 API 端点总结表

| 端点 | 方法 | 请求 | 响应 | 延迟 | 用途 |
|------|------|------|------|------|------|
| `/health` | GET | - | HealthResponse | <1ms | 健康检查 |
| `/embed/query` | POST | QueryRequest | EmbeddingResponse | ~100ms | 单文本嵌入 |
| `/embed/documents` | POST | DocumentsRequest | EmbeddingsResponse | ~500ms-2s | 批量嵌入 |
| `/chat` | POST | ChatRequest | ChatResponse | ~2-10s | 聊天补全 |

---

## 🎯 关键特点

### ✅ 优点

1. **简洁** - 仅 110 行代码
2. **自动文档** - FastAPI 自动生成 Swagger UI
3. **类型安全** - Pydantic 自动验证
4. **结构清晰** - 数据模型 + 工厂 + 路由分离
5. **易于测试** - 依赖注入，便于 Mock

### ⚠️ 改进空间

1. **错误处理** - 过于宽泛，需要更精细的分类
2. **日志缺失** - 无结构化日志输出
3. **性能优化** - 动态导入（/chat 端点）
4. **验证不足** - 无输入大小限制
5. **监控缺失** - 无性能指标收集
6. **超时处理** - 无超时设置
7. **速率限制** - 无限流控制

---

## 🔧 使用场景

### 场景 1: 本地开发

```bash
# 启动 Ollama
ollama serve

# 启动 API
PROVIDER=ollama EMBED_MODEL=mxbai-embed-large \
uvicorn embedding_service.api:app --reload

# 测试
curl http://localhost:8000/health
```

### 场景 2: Docker 部署

```bash
# Dockerfile
FROM python:3.11
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY embedding_service ./embedding_service
ENV PROVIDER=ollama
ENV OLLAMA_BASE_URL=http://ollama:11434
CMD ["uvicorn", "embedding_service.api:app", "--host", "0.0.0.0"]
```

### 场景 3: 生产部署

```yaml
# kubernetes deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: embedding-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: embedding-service
  template:
    metadata:
      labels:
        app: embedding-service
    spec:
      containers:
      - name: api
        image: embedding-service:latest
        ports:
        - containerPort: 8000
        env:
        - name: PROVIDER
          value: "ollama"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

---

## 📈 可扩展性

### 添加新端点

```python
# 在 create_app() 中添加:

class SimilarityRequest(BaseModel):
    vector1: List[float]
    vector2: List[float]

class SimilarityResponse(BaseModel):
    similarity: float

@app.post("/similarity", response_model=SimilarityResponse)
def similarity(request: SimilarityRequest) -> SimilarityResponse:
    """Calculate cosine similarity between two vectors."""
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    
    v1 = np.array(request.vector1).reshape(1, -1)
    v2 = np.array(request.vector2).reshape(1, -1)
    score = cosine_similarity(v1, v2)[0][0]
    return SimilarityResponse(similarity=float(score))
```

---

## 总结

**api.py** 是一个精简但功能完整的 REST API 实现，提供了 4 个核心端点来支持嵌入和聊天操作。它展示了以下最佳实践：

✅ 使用 Pydantic 进行数据验证  
✅ 使用工厂模式创建应用  
✅ 分离数据模型和业务逻辑  
✅ 基于类型提示的自动文档  

但在生产环境中需要添加：
- 更精细的错误处理
- 结构化日志
- 性能监控
- 速率限制
- 超时设置

