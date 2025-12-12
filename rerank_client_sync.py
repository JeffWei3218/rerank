"""
同步客户端测试脚本
用于验证 SyncRerankClient 是否正常工作
"""

def test_sync_client():
    """测试同步客户端"""
    try:
        # 尝试导入 requests
        import requests
    except ImportError:
        print("❌ 缺少 requests 库，请安装: pip install requests")
        return
    
    # 导入同步客户端
    from rerank_client import SyncRerankClient
    
    print("="*60)
    print("同步客户端测试")
    print("="*60 + "\n")
    
    # 初始化客户端
    client = SyncRerankClient(
        base_url="http://localhost:8000"
        # api_key="your-api-key"  # 如果需要
    )
    
    try:
        # 1. 健康检查
        print("1️⃣ 检查服务状态...")
        if not client.health_check():
            print("❌ Rerank 服务未运行，请先启动服务")
            return
        print("✅ 服务运行正常\n")
        
        # 2. 基础测试
        print("2️⃣ 测试基础重排...")
        query = "什么是机器学习？"
        documents = [
            "机器学习是人工智能的一个分支",
            "Python 是一种编程语言",
            "深度学习使用神经网络"
        ]
        
        results = client.rerank(
            query=query,
            documents=documents,
            top_n=2
        )
        
        print(f"查询: {query}")
        print(f"返回结果数: {len(results)}\n")
        
        for i, result in enumerate(results, 1):
            print(f"排名 {i}:")
            print(f"  索引: {result.index}")
            print(f"  分数: {result.relevance_score:.4f}")
            print(f"  文档: {documents[result.index]}\n")
        
        # 3. 测试不同模型
        print("3️⃣ 测试指定模型...")
        results_base = client.rerank(
            query=query,
            documents=documents,
            top_n=1,
            model="BAAI/bge-reranker-base"
        )
        
        print(f"使用 base 模型:")
        print(f"  Top1 索引: {results_base[0].index}")
        print(f"  Top1 分数: {results_base[0].relevance_score:.4f}\n")
        
        # 4. 测试完整文档列表
        print("4️⃣ 测试大量文档...")
        many_docs = [f"文档{i}的内容关于机器学习" for i in range(20)]
        many_docs[10] = "机器学习是让计算机从数据中学习的技术"
        
        results_many = client.rerank(
            query="机器学习",
            documents=many_docs,
            top_n=3
        )
        
        print(f"处理 {len(many_docs)} 个文档，返回 Top {len(results_many)}:")
        for i, result in enumerate(results_many, 1):
            print(f"  {i}. 索引 {result.index}, 分数: {result.relevance_score:.4f}")
        
        print("\n" + "="*60)
        print("✅ 所有测试通过！")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        client.close()


def test_error_handling():
    """测试错误处理"""
    from rerank_client import SyncRerankClient
    
    print("\n" + "="*60)
    print("错误处理测试")
    print("="*60 + "\n")
    
    # 测试连接错误的服务
    client = SyncRerankClient(base_url="http://localhost:9999")
    
    print("1️⃣ 测试连接不存在的服务...")
    if client.health_check():
        print("⚠️  意外：服务存在")
    else:
        print("✅ 正确检测到服务不可用\n")
    
    # 测试空文档列表
    client = SyncRerankClient(base_url="http://localhost:8000")
    
    print("2️⃣ 测试空文档列表...")
    try:
        results = client.rerank(
            query="测试",
            documents=[],
            top_n=1
        )
        print("⚠️  应该抛出异常")
    except Exception as e:
        print(f"✅ 正确捕获异常: {type(e).__name__}\n")
    
    print("="*60)
    print("✅ 错误处理测试完成！")
    print("="*60)


def compare_async_vs_sync():
    """对比异步和同步客户端"""
    import asyncio
    import time
    from rerank_client import RerankClient, SyncRerankClient
    
    print("\n" + "="*60)
    print("异步 vs 同步性能对比")
    print("="*60 + "\n")
    
    query = "机器学习"
    documents = [f"文档{i}" for i in range(50)]
    
    # 测试同步客户端
    print("1️⃣ 测试同步客户端...")
    sync_client = SyncRerankClient()
    
    start = time.time()
    for _ in range(5):
        sync_client.rerank(query, documents, top_n=5)
    sync_time = time.time() - start
    
    print(f"同步客户端 5 次请求耗时: {sync_time:.3f}秒")
    print(f"平均每次: {sync_time/5:.3f}秒\n")
    
    sync_client.close()
    
    # 测试异步客户端（串行）
    print("2️⃣ 测试异步客户端（串行）...")
    
    async def test_async_serial():
        client = RerankClient()
        try:
            start = time.time()
            for _ in range(5):
                await client.rerank(query, documents, top_n=5)
            return time.time() - start
        finally:
            await client.close()
    
    async_serial_time = asyncio.run(test_async_serial())
    print(f"异步客户端（串行）5 次请求耗时: {async_serial_time:.3f}秒")
    print(f"平均每次: {async_serial_time/5:.3f}秒\n")
    
    # 测试异步客户端（并发）
    print("3️⃣ 测试异步客户端（并发）...")
    
    async def test_async_concurrent():
        client = RerankClient()
        try:
            start = time.time()
            tasks = [
                client.rerank(query, documents, top_n=5)
                for _ in range(5)
            ]
            await asyncio.gather(*tasks)
            return time.time() - start
        finally:
            await client.close()
    
    async_concurrent_time = asyncio.run(test_async_concurrent())
    print(f"异步客户端（并发）5 次请求耗时: {async_concurrent_time:.3f}秒")
    print(f"平均每次: {async_concurrent_time/5:.3f}秒\n")
    
    print("="*60)
    print("性能总结:")
    print(f"  同步客户端:         {sync_time:.3f}秒 (基线)")
    print(f"  异步客户端（串行）: {async_serial_time:.3f}秒 ({async_serial_time/sync_time*100:.1f}%)")
    print(f"  异步客户端（并发）: {async_concurrent_time:.3f}秒 ({async_concurrent_time/sync_time*100:.1f}%)")
    print(f"\n💡 并发加速比: {sync_time/async_concurrent_time:.2f}x")
    print("="*60)


if __name__ == "__main__":
    # 运行测试
    test_sync_client()
    
    # 错误处理测试
    test_error_handling()
    
    # 性能对比（可选）
    try:
        compare_async_vs_sync()
    except Exception as e:
        print(f"\n⚠️  性能对比跳过: {e}")