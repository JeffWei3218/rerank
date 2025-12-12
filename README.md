# FastAPI Rerank API 服务

> 兼容 VLLM 格式的本地 Rerank API 服务，支持多模型动态加载

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 📖 目录

- [功能特性](#功能特性)
- [快速开始](#快速开始)
- [安装部署](#安装部署)
- [API 使用](#api-使用)
- [多模型支持](#多模型支持)
- [客户端集成](#客户端集成)
- [配置选项](#配置选项)
- [性能优化](#性能优化)
- [故障排查](#故障排查)
- [最佳实践](#最佳实践)

## ✨ 功能特性

- 🚀 **即插即用** - 兼容 VLLM Rerank Provider，客户端无需修改
- 🎯 **多模型支持** - 支持动态加载多个 Rerank 模型
- 📂 **本地优先** - 优先使用本地模型，支持离线部署
- 🔐 **安全认证** - 支持 API Key 认证（可选）
- ⚡ **高性能** - 模型缓存、异步处理
- 📊 **智能管理** - 按需加载、内存优化
- 🌏 **中文优化** - 基于 BGE Reranker 系列模型

## 🚀 快速开始

### 1. 安装依赖

```bash
# 克隆或下载项目
git clone <your-repo-url>
cd rerank-api

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install fastapi uvicorn sentence-transformers aiohttp
```

### 2. 下载模型

```bash
# 使用一键下载脚本（推荐）
python download_model.py

# 按提示选择模型：
# 1. BAAI/bge-reranker-large（推荐，高精度）
# 2. BAAI/bge-reranker-base（快速，平衡）
# 3. BAAI/bge-reranker-v2-m3（多语言）
```

**中国大陆用户加速：**
```bash
export HF_ENDPOINT=https://hf-mirror.com
python download_model.py
```

### 3. 启动服务

```bash
# 基础启动
python rerank_server.py

# 带 API Key 认证（生产环境推荐）
export RERANK_API_KEY="your-secret-key"
python rerank_server.py

# 自定义端口
uvicorn rerank_server:app --port 8001
```

### 4. 测试服务

```bash
# 健康检查
curl http://localhost:8000/

# 测试重排
curl -X POST "http://localhost:8000/v1/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "什么是机器学习？",
    "documents": ["机器学习是AI的分支", "Python是编程语言"],
    "top_n": 1
  }'
```

## 📦 安装部署

### 目录结构

```
rerank-api/
├── rerank_server.py          # 服务端主程序
├── rerank_client.py          # 客户端示例
├── download_model.py         # 模型下载脚本
├── README.md                 # 本文档
├── requirements.txt          # 依赖列表
└── models/                   # 模型目录
    ├── bge-reranker-base/
    ├── bge-reranker-large/
    └── bge-reranker-v2-m3/
```

### 依赖要求

创建 `requirements.txt`：

```txt
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
sentence-transformers>=2.2.0
aiohttp>=3.8.0
torch>=2.0.0
requests>=2.28.0  # 同步客户端需要
```

安装：
```bash
pip install -r requirements.txt
```

### 模型准备（4 种方式）

**方式 1：使用一键下载脚本（推荐）**
```bash
python download_model.py
```

**方式 2：Python 脚本下载**
```python
from sentence_transformers import CrossEncoder
model = CrossEncoder('BAAI/bge-reranker-large')
model.save('models/bge-reranker-large')
```

**方式 3：使用 huggingface-cli**
```bash
pip install huggingface_hub
huggingface-cli download BAAI/bge-reranker-large --local-dir models/bge-reranker-large
```

**方式 4：手动下载**
访问 [HuggingFace](https://huggingface.co/BAAI/bge-reranker-large/tree/main) 下载所有文件到 `models/bge-reranker-large/`

## 📝 API 使用

### 服务端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 健康检查 |
| `/v1/rerank` | POST | 重排文档 |
| `/v1/models` | GET | 列出支持的模型 |
| `/docs` | GET | Swagger 文档 |

### 请求格式

**POST /v1/rerank**

```json
{
  "query": "你的查询文本",
  "documents": ["文档1", "文档2", "文档3"],
  "model": "BAAI/bge-reranker-base",  // 可选，默认 large
  "top_n": 2  // 可选，返回前 n 个结果
}
```

### 响应格式

```json
{
  "results": [
    {
      "index": 1,
      "relevance_score": 0.9856
    },
    {
      "index": 0,
      "relevance_score": 0.7234
    }
  ]
}
```

### cURL 示例

```bash
# 基础调用
curl -X POST "http://localhost:8000/v1/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "什么是深度学习？",
    "documents": [
      "深度学习使用多层神经网络",
      "Python是编程语言",
      "机器学习是AI的分支"
    ],
    "top_n": 2
  }'

# 使用 API Key
curl -X POST "http://localhost:8000/v1/rerank" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{"query": "...", "documents": [...]}'

# 指定模型
curl -X POST "http://localhost:8000/v1/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "你的查询",
    "documents": ["文档1", "文档2"],
    "model": "BAAI/bge-reranker-base"
  }'
```

## 🎯 多模型支持

### 支持的模型

| 模型名称 | 本地路径 | 大小 | 特点 | 推荐场景 |
|---------|---------|------|------|---------|
| `BAAI/bge-reranker-base` | `models/bge-reranker-base` | ~400MB | 快速、平衡 | 生产环境，低延迟 |
| `BAAI/bge-reranker-large` | `models/bge-reranker-large` | ~1.1GB | 高精度（默认） | 对准确率要求高 |
| `BAAI/bge-reranker-v2-m3` | `models/bge-reranker-v2-m3` | ~560MB | 多语言优化 | 跨语言检索 |

### 动态加载机制

```
客户端请求 model=bge-reranker-base
         ↓
    检查是否已加载？
    ├─ 是 → 直接使用缓存的模型
    └─ 否 → 加载流程：
            1. 检查本地是否有模型文件
            2. 有 → 从本地加载
            3. 无 → 从 HuggingFace 下载
            4. 缓存到内存
            5. 返回结果
```

### 查看模型状态

```bash
# 查看已加载的模型
curl http://localhost:8000/

# 响应示例
{
  "status": "running",
  "loaded_models": ["BAAI/bge-reranker-large"],
  "default_model": "BAAI/bge-reranker-large",
  "supported_models": [
    "BAAI/bge-reranker-base",
    "BAAI/bge-reranker-large",
    "BAAI/bge-reranker-v2-m3"
  ]
}

# 列出所有模型详情
curl http://localhost:8000/v1/models
```

### 使用不同模型

```python
# 方式 1: 使用默认模型（无需指定）
results = await client.rerank(
    query="你的查询",
    documents=["文档1", "文档2"]
)

# 方式 2: 指定使用 base 模型（更快）
results = await client.rerank(
    query="你的查询",
    documents=["文档1", "文档2"],
    model="BAAI/bge-reranker-base"
)

# 方式 3: 使用多语言模型
results = await client.rerank(
    query="What is AI?",
    documents=["Doc1", "Doc2"],
    model="BAAI/bge-reranker-v2-m3"
)
```

## 🔌 客户端集成

### 使用提供的异步客户端

```python
import asyncio
from rerank_client import RerankClient

async def main():
    # 初始化客户端
    client = RerankClient(
        base_url="http://localhost:8000",
        api_key="your-api-key"  # 可选
    )
    
    try:
        # 调用 Rerank
        results = await client.rerank(
            query="什么是机器学习？",
            documents=[
                "机器学习是AI的分支",
                "Python是编程语言",
                "深度学习使用神经网络"
            ],
            top_n=2
        )
        
        # 处理结果
        for result in results:
            print(f"索引: {result.index}, 分数: {result.relevance_score:.4f}")
    
    finally:
        await client.close()

asyncio.run(main())
```

### 使用同步客户端

```python
from rerank_client import SyncRerankClient

# 初始化
client = SyncRerankClient(
    base_url="http://localhost:8000",
    api_key="your-api-key"
)

# 调用
results = client.rerank(
    query="你的查询",
    documents=["文档1", "文档2"],
    top_n=2
)

# 处理结果
for result in results:
    print(f"索引: {result.index}, 分数: {result.relevance_score:.4f}")

client.close()
```

### 与 VLLMRerankProvider 集成

**完全兼容，无需修改代码！**

```python
from your_module import VLLMRerankProvider

# 配置
provider_config = {
    "rerank_api_base": "http://127.0.0.1:8000",
    "rerank_api_key": "your-api-key",  # 可选
    "rerank_model": "BAAI/bge-reranker-large",
    "timeout": 20
}

# 初始化
provider = VLLMRerankProvider(provider_config, {})

# 使用（无需修改）
results = await provider.rerank(
    query="你的查询",
    documents=["文档1", "文档2"],
    top_n=2
)

for r in results:
    print(f"索引: {r.index}, 分数: {r.relevance_score}")
```

### RAG 应用集成

```python
import asyncio
from rerank_client import RerankClient

async def rag_pipeline(user_query: str):
    """完整的 RAG 流程"""
    reranker = RerankClient("http://localhost:8000")
    
    try:
        # 1. 向量检索（召回候选）
        candidates = await vector_search(user_query, top_k=20)
        
        # 2. Rerank 精排
        reranked = await reranker.rerank(
            query=user_query,
            documents=candidates,
            top_n=5  # 只保留最相关的 5 个
        )
        
        # 3. 提取最相关的文档
        top_docs = [candidates[r.index] for r in reranked]
        
        # 4. 生成答案
        answer = await llm_generate(user_query, top_docs)
        
        return answer
    
    finally:
        await reranker.close()

# 使用
answer = asyncio.run(rag_pipeline("如何提高模型性能？"))
```

## ⚙️ 配置选项

### 环境变量

```bash
# API Key（可选，生产环境推荐）
export RERANK_API_KEY="your-secret-key"

# HuggingFace 镜像（中国大陆用户）
export HF_ENDPOINT=https://hf-mirror.com
```

### 修改默认端口

```python
# rerank_server.py 最后一行
uvicorn.run(app, host="0.0.0.0", port=8001)  # 改为 8001

# 或使用命令行
uvicorn rerank_server:app --port 8001
```

### 修改默认模型

```python
# rerank_server.py 中的 load_model 函数
@app.on_event("startup")
async def load_model():
    global rerank_models, default_model_name
    
    # 改为 base 作为默认模型（更快）
    default_model_name = "BAAI/bge-reranker-base"
    rerank_models[default_model_name] = load_single_model(default_model_name)
```

### 预加载多个模型

```python
# 避免首次请求延迟
@app.on_event("startup")
async def load_model():
    global rerank_models, default_model_name
    
    # 预加载多个常用模型
    models_to_preload = [
        "BAAI/bge-reranker-large",
        "BAAI/bge-reranker-base"
    ]
    
    for model_name in models_to_preload:
        logger.info(f"预加载模型: {model_name}")
        rerank_models[model_name] = load_single_model(model_name)
    
    default_model_name = "BAAI/bge-reranker-large"
```

### 添加新模型

```python
# 编辑 SUPPORTED_MODELS 字典
SUPPORTED_MODELS = {
    "BAAI/bge-reranker-base": {...},
    "BAAI/bge-reranker-large": {...},
    "BAAI/bge-reranker-v2-m3": {...},
    # 添加新模型
    "cross-encoder/ms-marco-MiniLM-L-6-v2": {
        "local_path": "models/ms-marco-minilm",
        "remote_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "max_length": 512
    }
}
```

## 🚀 性能优化

### 性能对比

#### 模型性能

| 模型 | CPU 推理 | GPU 推理 | 内存占用 | 精度 |
|------|---------|---------|---------|------|
| base | ~200ms | ~50ms | ~400MB | ⭐⭐⭐ |
| large | ~400ms | ~100ms | ~1.2GB | ⭐⭐⭐⭐⭐ |
| v2-m3 | ~250ms | ~70ms | ~600MB | ⭐⭐⭐⭐ |

*测试条件：100 个文档，Intel i7 / RTX 3080*

#### 加载时间

| 场景 | 本地加载 | 网络下载 |
|------|---------|---------|
| 首次启动 | 3-5秒 | 60-120秒 |
| 动态加载新模型 | 2-3秒 | 30-80秒 |
| 使用缓存模型 | <10ms | - |

### GPU 加速

```bash
# 安装 GPU 版本 PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu118

# 模型会自动使用 CUDA
# 推理速度提升 3-5 倍
```

### 批量处理

```python
# 并发处理多个查询
import asyncio

async def batch_rerank(queries, documents):
    client = RerankClient()
    
    # 创建并发任务
    tasks = [
        client.rerank(query, documents, top_n=3)
        for query in queries
    ]
    
    # 并发执行
    results = await asyncio.gather(*tasks)
    
    await client.close()
    return results

# 使用
queries = ["查询1", "查询2", "查询3"]
results = asyncio.run(batch_rerank(queries, documents))
```

### 缓存策略

```python
# 对常见查询缓存结果
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_rerank(query: str, docs_hash: str):
    # 实现缓存逻辑
    pass
```

## 🐳 Docker 部署

### Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY rerank_server.py .
COPY models/ ./models/

# 暴露端口
EXPOSE 8000

# 启动服务
CMD ["python", "rerank_server.py"]
```

### 构建和运行

```bash
# 构建镜像
docker build -t rerank-api .

# 运行容器
docker run -d \
  -p 8000:8000 \
  -e RERANK_API_KEY=your-secret-key \
  --name rerank-api \
  rerank-api

# 查看日志
docker logs -f rerank-api
```

### Docker Compose

```yaml
version: '3.8'

services:
  rerank-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - RERANK_API_KEY=your-secret-key
    volumes:
      - ./models:/app/models
    restart: unless-stopped
```

## 🔍 故障排查

### 问题 1：模型下载失败

**症状：** 启动时提示网络错误

**解决方案：**
```bash
# 使用镜像站点
export HF_ENDPOINT=https://hf-mirror.com
python download_model.py

# 或手动下载后放入 models/ 目录
```

### 问题 2：内存不足

**症状：** 加载模型时 OOM 错误

**解决方案：**
```python
# 使用更小的模型
default_model_name = "BAAI/bge-reranker-base"

# 或只加载必需的模型
# 不要预加载多个大模型
```

### 问题 3：端口被占用

**症状：** Address already in use

**解决方案：**
```bash
# 查看端口占用
lsof -i :8000  # Mac/Linux
netstat -ano | findstr :8000  # Windows

# 更换端口
uvicorn rerank_server:app --port 8001
```

### 问题 4：API 认证失败

**症状：** 401 Unauthorized

**解决方案：**
```bash
# 确认环境变量已设置
echo $RERANK_API_KEY

# 客户端确保传入正确的 API Key
client = RerankClient(api_key="your-api-key")
```

### 问题 5：模型加载很慢

**症状：** 首次请求耗时很长

**解决方案：**
```bash
# 提前下载模型到本地
python download_model.py

# 或在 startup 中预加载
# 见"预加载多个模型"配置
```

### 问题 6：同步客户端运行错误

**症状：** `RuntimeError: no running event loop`

**原因：** 旧版同步客户端在非异步环境中创建 aiohttp session 导致

**解决方案：**
```bash
# 1. 确保安装了 requests 库
pip install requests

# 2. 使用新版同步客户端（基于 requests）
from rerank_client import SyncRerankClient

client = SyncRerankClient()
results = client.rerank(query, documents)
```

新版同步客户端使用 `requests` 库而非 `aiohttp`，避免了事件循环问题。

## 💡 最佳实践

### 1. 按场景选择模型

```python
# 实时对话系统：优先速度
model = "BAAI/bge-reranker-base"

# 知识库问答：优先准确率
model = "BAAI/bge-reranker-large"

# 多语言场景
model = "BAAI/bge-reranker-v2-m3"
```

### 2. RAG 两阶段检索

```python
# 第一阶段：向量检索（召回）
candidates = vector_search(query, top_k=50)  # 召回 50 个

# 第二阶段：Rerank 精排
reranked = rerank(query, candidates, top_n=5)  # 精排到 5 个

# 使用精排后的结果生成答案
answer = generate(query, reranked)
```

### 3. 生产环境部署

```bash
# 1. 启用 API Key 认证
export RERANK_API_KEY="strong-random-key"

# 2. 使用本地模型（避免下载延迟）
python download_model.py

# 3. 预加载常用模型
# 编辑 rerank_server.py 的 startup 函数

# 4. 使用进程管理器
pip install supervisor
# 配置 supervisord.conf

# 5. 配置反向代理（可选）
# nginx 反向代理到 8000 端口
```

### 4. 监控和日志

```python
# 添加日志记录
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rerank.log'),
        logging.StreamHandler()
    ]
)

# 监控请求耗时
import time
start = time.time()
results = await client.rerank(query, docs)
print(f"耗时: {time.time() - start:.2f}秒")
```

### 5. 错误处理

```python
import asyncio
from rerank_client import RerankClient

async def safe_rerank(query, documents):
    client = RerankClient()
    
    try:
        results = await client.rerank(query, documents, top_n=5)
        return results
    except asyncio.TimeoutError:
        print("请求超时，使用默认排序")
        return [{"index": i, "relevance_score": 0} for i in range(len(documents))]
    except Exception as e:
        print(f"Rerank 失败: {e}")
        return []
    finally:
        await client.close()
```

## 📊 性能基准测试

### 测试脚本

```python
import asyncio
import time
from rerank_client import RerankClient

async def benchmark():
    client = RerankClient()
    
    query = "机器学习算法"
    documents = [f"文档{i}" for i in range(100)]
    
    # 预热
    await client.rerank(query, documents[:10])
    
    # 测试
    times = []
    for _ in range(10):
        start = time.time()
        await client.rerank(query, documents)
        times.append(time.time() - start)
    
    print(f"平均耗时: {sum(times)/len(times):.3f}秒")
    print(f"QPS: {1/(sum(times)/len(times)):.2f}")
    
    await client.close()

asyncio.run(benchmark())
```

## 📄 License

MIT License - 详见 [LICENSE](LICENSE) 文件

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📞 支持

- 📧 Email: your-email@example.com
- 💬 Issues: [GitHub Issues](https://github.com/your-repo/issues)
- 📖 文档: [完整文档](https://your-docs-site.com)

## 🙏 致谢

- [BAAI/bge-reranker](https://huggingface.co/BAAI/bge-reranker-large) - 优秀的 Rerank 模型
- [FastAPI](https://fastapi.tiangolo.com/) - 现代化的 Web 框架
- [sentence-transformers](https://www.sbert.net/) - 强大的句子嵌入库

## 📝 更新日志

### v1.0.0 (2024-01-XX)
- ✨ 初始版本发布
- 🎯 支持多模型动态加载
- 🔐 支持 API Key 认证
- 📂 支持本地模型优先加载
- 🚀 兼容 VLLM Rerank Provider

---

**⭐ 如果这个项目对你有帮助，请给个 Star！**