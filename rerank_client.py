import asyncio
import aiohttp
from typing import List, Optional

class RerankResult:
    """重排结果"""
    def __init__(self, index: int, relevance_score: float):
        self.index = index
        self.relevance_score = relevance_score
    
    def __repr__(self):
        return f"RerankResult(index={self.index}, score={self.relevance_score:.4f})"

class RerankClient:
    """
    Rerank API 异步客户端（兼容 VLLM 格式）
    """
    
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        api_key: Optional[str] = None,
        timeout: int = 20
    ):
        """
        初始化客户端
        
        Args:
            base_url: API 基础 URL
            api_key: API Key（可选）
            timeout: 请求超时时间（秒）
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        
        # 构建请求头
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        self.session = aiohttp.ClientSession(
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
    
    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
        model: str = "BAAI/bge-reranker-base"
    ) -> List[RerankResult]:
        """
        重排文档
        
        Args:
            query: 查询文本
            documents: 待重排的文档列表
            top_n: 返回前 n 个结果
            model: 模型名称
        
        Returns:
            重排结果列表
        """
        payload = {
            "query": query,
            "documents": documents,
            "model": model
        }
        
        if top_n is not None:
            payload["top_n"] = top_n
        
        try:
            async with self.session.post(
                f"{self.base_url}/v1/rerank",
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API 错误 {response.status}: {error_text}")
                
                data = await response.json()
                results = data.get("results", [])
                
                if not results:
                    print(f"⚠️  API 返回了空结果。原始响应: {data}")
                
                return [
                    RerankResult(
                        index=r["index"],
                        relevance_score=r["relevance_score"]
                    )
                    for r in results
                ]
        
        except Exception as e:
            print(f"❌ Rerank 请求失败: {e}")
            raise
    
    async def health_check(self) -> bool:
        """检查服务是否运行"""
        try:
            async with self.session.get(f"{self.base_url}/") as response:
                return response.status == 200
        except:
            return False
    
    async def close(self):
        """关闭客户端会话"""
        if self.session:
            await self.session.close()
            self.session = None


# ============== 使用示例 ==============

async def basic_example():
    """基础使用示例"""
    print("="*60)
    print("基础重排示例")
    print("="*60 + "\n")
    
    # 初始化客户端（如果需要 API Key，传入 api_key 参数）
    client = RerankClient(
        base_url="http://localhost:8000",
        # api_key="your-api-key-here"  # 如果服务端设置了 API Key
    )
    
    try:
        # 检查服务状态
        if not await client.health_check():
            print("❌ Rerank 服务未运行")
            return
        
        print("✅ Rerank 服务运行正常\n")
        
        # 准备测试数据
        query = "什么是机器学习中的过拟合？"
        documents = [
            "深度学习是机器学习的一个分支，使用多层神经网络。",
            "过拟合是指模型在训练数据上表现很好，但在新数据上表现差。",
            "Python 是一种流行的编程语言。",
            "正则化是防止过拟合的常用技术。",
            "卷积神经网络主要用于图像处理。"
        ]
        
        print(f"查询: {query}\n")
        print(f"文档数量: {len(documents)}\n")
        
        # 调用 Rerank（使用默认模型）
        print("使用默认模型（bge-reranker-large）...")
        results = await client.rerank(
            query=query,
            documents=documents,
            top_n=3
        )
        
        # 显示结果
        print(f"\n重排结果（Top {len(results)}）:\n")
        for i, result in enumerate(results, 1):
            print(f"排名 {i}:")
            print(f"  原始索引: {result.index}")
            print(f"  相关性分数: {result.relevance_score:.4f}")
            print(f"  文档内容: {documents[result.index]}\n")
        
        # 尝试使用不同模型
        print("\n" + "="*60)
        print("使用 base 模型（更快）...")
        results_base = await client.rerank(
            query=query,
            documents=documents,
            top_n=3,
            model="BAAI/bge-reranker-base"
        )
        print(f"✅ Base 模型返回 {len(results_base)} 个结果")
        print(f"Top1 分数: {results_base[0].relevance_score:.4f}")
    
    finally:
        await client.close()


async def rag_example():
    """RAG 应用集成示例"""
    print("="*60)
    print("RAG 集成示例")
    print("="*60 + "\n")
    
    client = RerankClient()
    
    try:
        # 模拟 RAG 流程
        user_query = "如何提高深度学习模型的泛化能力？"
        
        # 假设从向量数据库检索到的候选文档（召回阶段）
        candidates = [
            "数据增强可以增加训练数据的多样性。",
            "Dropout 是一种常用的正则化技术。",
            "梯度下降是优化算法的基础。",
            "Early Stopping 可以防止过拟合。",
            "批归一化有助于训练稳定。",
            "交叉验证用于评估模型性能。",
            "L2 正则化通过惩罚大权重来防止过拟合。",
            "集成学习可以提高模型鲁棒性。"
        ]
        
        print(f"步骤 1: 向量检索 - 召回 {len(candidates)} 个候选文档")
        print(f"步骤 2: Rerank 精排...")
        
        # 使用 Rerank 精排
        results = await client.rerank(
            query=user_query,
            documents=candidates,
            top_n=3  # 只保留最相关的 3 个
        )
        
        print(f"步骤 3: 精排完成，选取 Top {len(results)} 文档\n")
        
        # 显示最终选择的文档
        print("精排后的文档（用于生成答案）:\n")
        for i, result in enumerate(results, 1):
            print(f"{i}. [分数: {result.relevance_score:.4f}] {candidates[result.index]}")
        
        print("\n步骤 4: 使用这些文档生成最终答案（调用 LLM）")
    
    finally:
        await client.close()


async def batch_example():
    """批量处理示例"""
    print("="*60)
    print("批量处理示例")
    print("="*60 + "\n")
    
    client = RerankClient()
    
    try:
        # 多个查询
        queries = [
            "什么是机器学习？",
            "如何防止过拟合？",
            "神经网络的优化方法"
        ]
        
        documents = [
            "机器学习是人工智能的一个分支。",
            "过拟合可以通过正则化解决。",
            "梯度下降是常用的优化算法。",
            "深度学习使用多层神经网络。"
        ]
        
        print(f"批量处理 {len(queries)} 个查询...\n")
        
        # 并发处理多个查询
        tasks = [
            client.rerank(query, documents, top_n=2)
            for query in queries
        ]
        
        results_list = await asyncio.gather(*tasks)
        
        # 显示结果
        for query, results in zip(queries, results_list):
            print(f"查询: {query}")
            print(f"最相关文档: {documents[results[0].index]}")
            print(f"相关性分数: {results[0].relevance_score:.4f}\n")
    
    finally:
        await client.close()


# 同步包装器（适配非异步环境）
class SyncRerankClient:
    """同步版本的 Rerank 客户端（使用 requests）"""
    
    def __init__(
        self, 
        base_url: str = "http://127.0.0.1:8000", 
        api_key: Optional[str] = None,
        timeout: int = 20
    ):
        try:
            import requests
            self.requests = requests
        except ImportError:
            raise ImportError("同步客户端需要安装 requests: pip install requests")
        
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        
        # 构建请求头
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
    
    def rerank(
        self, 
        query: str, 
        documents: List[str], 
        top_n: Optional[int] = None,
        model: str = "BAAI/bge-reranker-base"
    ) -> List[RerankResult]:
        """同步版本的 rerank 方法"""
        payload = {
            "query": query,
            "documents": documents,
            "model": model
        }
        
        if top_n is not None:
            payload["top_n"] = top_n
        
        try:
            response = self.requests.post(
                f"{self.base_url}/v1/rerank",
                json=payload,
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            results = data.get("results", [])
            
            return [
                RerankResult(
                    index=r["index"],
                    relevance_score=r["relevance_score"]
                )
                for r in results
            ]
        
        except Exception as e:
            print(f"❌ Rerank 请求失败: {e}")
            raise
    
    def health_check(self) -> bool:
        """检查服务是否运行"""
        try:
            response = self.requests.get(f"{self.base_url}/", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def close(self):
        """关闭会话（requests 不需要显式关闭）"""
        pass


# ============== 主函数 ==============

async def main():
    """运行所有示例"""
    await basic_example()
    print("\n")
    await rag_example()
    print("\n")
    await batch_example()


if __name__ == "__main__":
    # 运行异步示例
    asyncio.run(main())
    
    # 同步使用示例
    print("\n" + "="*60)
    print("同步客户端示例")
    print("="*60 + "\n")
    
    sync_client = SyncRerankClient()
    results = sync_client.rerank(
        "机器学习",
        ["深度学习使用神经网络", "Python是编程语言"],
        top_n=1
    )
    print(f"最相关文档索引: {results[0].index}, 分数: {results[0].relevance_score:.4f}")
    sync_client.close()