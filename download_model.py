"""
下载 Rerank 模型到本地目录
支持多种模型选择和断点续传
"""

import os
import sys
from pathlib import Path

def download_model(model_name: str, save_dir: str):
    """
    下载模型到指定目录
    
    Args:
        model_name: Hugging Face 模型名称
        save_dir: 保存目录
    """
    try:
        from sentence_transformers import CrossEncoder
        
        print(f"📦 开始下载模型: {model_name}")
        print(f"💾 保存路径: {save_dir}")
        print("⏳ 请稍候，这可能需要几分钟...\n")
        
        # 创建目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 下载模型
        model = CrossEncoder(model_name, max_length=512)
        
        # 保存到本地
        model.save(save_dir)
        
        print(f"\n✅ 模型下载完成！")
        print(f"📂 模型文件位置: {os.path.abspath(save_dir)}")
        
        # 验证文件
        files = list(Path(save_dir).glob("*"))
        print(f"\n📋 包含 {len(files)} 个文件:")
        for f in files[:10]:  # 只显示前10个
            print(f"   - {f.name}")
        if len(files) > 10:
            print(f"   ... 还有 {len(files) - 10} 个文件")
            
        return True
        
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        print("\n💡 解决方案:")
        print("1. 检查网络连接")
        print("2. 使用镜像站点: export HF_ENDPOINT=https://hf-mirror.com")
        print("3. 尝试手动下载: https://huggingface.co/" + model_name)
        return False


def main():
    """主函数：提供交互式选择"""
    
    print("="*60)
    print("🤖 Rerank 模型下载工具")
    print("="*60 + "\n")
    
    # 模型选项
    models = {
        "1": {
            "name": "BAAI/bge-reranker-large",
            "dir": "models/bge-reranker-large",
            "desc": "BGE Reranker Large（推荐，高精度，约 1.1GB）"
        },
        "2": {
            "name": "BAAI/bge-reranker-base",
            "dir": "models/bge-reranker-base",
            "desc": "BGE Reranker Base（快速，约 400MB）"
        },
        "3": {
            "name": "BAAI/bge-reranker-v2-m3",
            "dir": "models/bge-reranker-v2-m3",
            "desc": "BGE Reranker v2 M3（多语言，约 560MB）"
        },
        "4": {
            "name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "dir": "models/ms-marco-minilm",
            "desc": "MS MARCO MiniLM（英文优化，约 80MB）"
        }
    }
    
    print("请选择要下载的模型：\n")
    for key, info in models.items():
        print(f"{key}. {info['desc']}")
        print(f"   模型: {info['name']}")
        print(f"   路径: {info['dir']}\n")
    
    # 获取用户选择
    choice = input("请输入选项 (1-4) [默认: 1]: ").strip() or "1"
    
    if choice not in models:
        print("❌ 无效选项！")
        return
    
    selected = models[choice]
    
    # 检查目录是否已存在
    if os.path.exists(selected['dir']):
        print(f"\n⚠️  目录已存在: {selected['dir']}")
        overwrite = input("是否覆盖？(y/n) [n]: ").strip().lower()
        if overwrite != 'y':
            print("已取消下载")
            return
    
    # 下载模型
    success = download_model(selected['name'], selected['dir'])
    
    if success:
        print("\n" + "="*60)
        print("🎉 完成！现在可以启动 Rerank API 服务了：")
        print("="*60)
        print("\npython rerank_server.py\n")


if __name__ == "__main__":
    # 检查依赖
    try:
        import sentence_transformers
    except ImportError:
        print("❌ 缺少依赖包！请先安装：")
        print("pip install sentence-transformers")
        sys.exit(1)
    
    main()