import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.chunking import ChunkingStrategyComparator

def main():
    comparator = ChunkingStrategyComparator()
    
    docs = [
        "data/ecommerce/chinh-sach-van-chuyen-shopee.md",
        "data/ecommerce/quy-che-hoat-dong-nguoi-ban-seller.md"
    ]
    
    for doc_path in docs:
        if not os.path.exists(doc_path):
            continue
        with open(doc_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        print(f"--- Tài liệu: {Path(doc_path).name} ---")
        results = comparator.compare(text, chunk_size=500)
        for strategy, stats in results.items():
            print(f"Chiến lược: {strategy}")
            print(f"Số lượng chunk: {stats['count']}")
            print(f"Độ dài trung bình: {stats['avg_length']}")
            # print a sample of the first chunk to judge context
            if stats['chunks']:
                sample = stats['chunks'][0].replace('\n', ' ')[:100]
                print(f"Mẫu chunk đầu: {sample}...")
            print("-" * 20)

if __name__ == "__main__":
    main()
