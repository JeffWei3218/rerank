from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn
from sentence_transformers import CrossEncoder
import logging
import os

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 创建 FastAPI 应用
app = FastAPI(
    title="VLLM Rerank API",
    description="兼容 VLLM 格式的 Rerank 服务",
    version="1.0.0"
)

# 全局变量
rerank_models = {}  # 模型缓存字典 {model_name: CrossEncoder}
default_model_name = None  # 默认模型名称
API_KEY = os.getenv("RERANK_API_KEY", "")  # 从环境变量读取 API Key

# 支持的模型配置
SUPPORTED_MODELS = {
    "BAAI/bge-reranker-base": {
        "local_path": "models/bge-reranker-base",
        "remote_name": "BAAI/bge-reranker-base",
        "max_length": 512
    },
    "BAAI/bge-reranker-large": {
        "local_path": "models/bge-reranker-large",
        "remote_name": "BAAI/bge-reranker-large",
        "max_length": 512
    },
    "BAAI/bge-reranker-v2-m3": {
        "local_path": "models/bge-reranker-v2-m3",
        "remote_name": "BAAI/bge-reranker-v2-m3",
        "max_length": 512
    }
}

# 请求模型（兼容 VLLM 格式）
class RerankRequest(BaseModel):
    query: str = Field(..., description="查询文本")
    documents: List[str] = Field(..., description="待重排的文档列表")
    model: Optional[str] = Field("BAAI/bge-reranker-base", description="模型名称（仅用于日志）")
    top_n: Optional[int] = Field(None, description="返回前 n 个结果")

# 响应模型（兼容 VLLM 格式）
class RerankResultItem(BaseModel):
    index: int = Field(..., description="文档在原始列表中的索引")
    relevance_score: float = Field(..., description="相关性分数")

class RerankResponse(BaseModel):
    results: List[RerankResultItem] = Field(..., description="重排结果列表")

# API Key 验证（可选）
async def verify_api_key(authorization: Optional[str] = Header(None)):
    """验证 API Key（如果设置了的话）"""
    if API_KEY:  # 只有设置了 API_KEY 才验证
        if not authorization:
            raise HTTPException(status_code=401, detail="未提供 Authorization header")
        
        # 支持 "Bearer <token>" 格式
        if authorization.startswith("Bearer "):
            token = authorization[7:]
        else:
            token = authorization
        
        if token != API_KEY:
            raise HTTPException(status_code=401, detail="无效的 API Key")
    
    return True

def load_single_model(model_name: str) -> CrossEncoder:
    """
    加载单个模型
    
    Args:
        model_name: 模型名称
    
    Returns:
        加载好的 CrossEncoder 模型
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"不支持的模型: {model_name}. 支持的模型: {list(SUPPORTED_MODELS.keys())}")
    
    config = SUPPORTED_MODELS[model_name]
    local_path = config["local_path"]
    remote_name = config["remote_name"]
    max_length = config["max_length"]
    
    # 优先使用本地模型
    if os.path.exists(local_path) and os.path.isdir(local_path):
        logger.info(f"✅ 发现本地模型: {local_path}")
        model_path = local_path
    else:
        logger.info(f"⚠️  本地模型不存在: {local_path}")
        logger.info(f"正在从 Hugging Face 下载: {remote_name}")
        model_path = remote_name
    
    model = CrossEncoder(model_path, max_length=max_length)
    logger.info(f"🎉 模型 [{model_name}] 加载成功！")
    
    return model


@app.on_event("startup")
async def load_model():
    """启动时加载默认模型"""
    global rerank_models, default_model_name
    try:
        # 默认加载 bge-reranker-large
        default_model_name = "BAAI/bge-reranker-large"
        
        logger.info(f"正在加载默认模型: {default_model_name}")
        rerank_models[default_model_name] = load_single_model(default_model_name)
        
        # 日志 API Key 状态
        if API_KEY:
            logger.info(f"🔐 API Key 认证已启用")
        else:
            logger.info(f"⚠️  未设置 API Key，服务无需认证（不推荐生产环境）")
        
        logger.info(f"✅ 服务启动完成！支持 {len(SUPPORTED_MODELS)} 个模型")
        
    except Exception as e:
        logger.error(f"❌ 模型加载失败: {str(e)}")
        raise

@app.get("/")
async def root():
    """健康检查端点"""
    return {
        "status": "running",
        "service": "VLLM Rerank API",
        "loaded_models": list(rerank_models.keys()),
        "default_model": default_model_name,
        "supported_models": list(SUPPORTED_MODELS.keys()),
        "authentication": "enabled" if API_KEY else "disabled"
    }

@app.post("/v1/rerank", response_model=RerankResponse)
async def rerank(
    request: RerankRequest,
    authorized: bool = Depends(verify_api_key)
):
    """
    重排文档接口（兼容 VLLM 格式）
    
    Args:
        request: 包含 query、documents 和可选参数
        authorized: API Key 验证结果
    
    Returns:
        重排后的文档列表（只包含 index 和 relevance_score）
    """
    global rerank_models, default_model_name
    
    if not request.documents:
        raise HTTPException(status_code=400, detail="文档列表不能为空")
    
    try:
        # 确定使用哪个模型
        model_name = request.model or default_model_name
        
        # 检查模型是否支持
        if model_name not in SUPPORTED_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的模型: {model_name}. 支持的模型: {list(SUPPORTED_MODELS.keys())}"
            )
        
        # 如果模型未加载，动态加载
        if model_name not in rerank_models:
            logger.info(f"🔄 模型 [{model_name}] 未加载，正在动态加载...")
            rerank_models[model_name] = load_single_model(model_name)
        
        model = rerank_models[model_name]
        
        logger.info(
            f"收到重排请求 - query: '{request.query[:50]}...', "
            f"documents: {len(request.documents)}个, "
            f"model: {model_name}, "
            f"top_n: {request.top_n}"
        )
        
        # 准备模型输入
        pairs = [[request.query, doc] for doc in request.documents]
        
        # 计算相关性分数
        scores = model.predict(pairs)
        
        # 创建结果列表
        results = [
            RerankResultItem(
                index=idx,
                relevance_score=float(score)
            )
            for idx, score in enumerate(scores)
        ]
        
        # 按分数降序排序
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # 如果指定了 top_n，只返回前 n 个
        if request.top_n is not None and request.top_n > 0:
            results = results[:request.top_n]
        
        # 记录最终返回的索引与分数
        try:
            return_scores_str = ", ".join([
                f"rank={i+1}->idx={r.index}: {r.relevance_score:.6f}"
                for i, r in enumerate(results)
            ]) or "(empty)"
            logger.info(
                f"✅ 重排完成，返回 {len(results)} 个结果（使用模型: {model_name}） - 返回列表: {return_scores_str}"
            )
        except Exception:
            logger.info(f"✅ 重排完成，返回 {len(results)} 个结果（使用模型: {model_name}）")
        
        return RerankResponse(results=results)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 重排失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"重排失败: {str(e)}")

@app.get("/v1/models")
async def list_models():
    """列出可用的模型"""
    return {
        "data": [
            {
                "id": model_name,
                "object": "model",
                "owned_by": "BAAI" if "BAAI" in model_name else "unknown",
                "loaded": model_name in rerank_models,
                "local_available": os.path.exists(config["local_path"])
            }
            for model_name, config in SUPPORTED_MODELS.items()
        ]
    }

if __name__ == "__main__":
    # 启动服务（默认端口 8000，兼容 VLLM）
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )