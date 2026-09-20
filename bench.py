#!/usr/bin/env python3
"""
Benchmark tool for evaluating retrieval quality on ecommerce corpus.
Day 7 — Data Foundations: Embedding & Vector Store
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from dotenv import load_dotenv

from src.models import Document
from src.store import EmbeddingStore
from src.chunking import (
    FixedSizeChunker,
    SentenceChunker,
    RecursiveChunker,
)
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    GEMINI_EMBEDDING_MODEL,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    GeminiEmbedder,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)

# 5 Benchmark Queries & Gold Answers — đồng bộ với report/REPORT_NHOM.md mục 3.
# gold_doc: doc_id THẬT trong data/ecommerce/ (đối chiếu với sources.csv, KHÔNG bịa hậu tố).
# gold_snippet: chuỗi đặc trưng phải có mặt trong ngữ cảnh top-3 mới tính là "trả lời được" —
#               chỉ khớp đúng gold_doc là chưa đủ (dễ bị thổi phồng, xem cảnh báo trong lab doc mục 7).
BENCHMARK_QUERIES = [
    {
        "id": 1,
        "query": "Người mua có bao nhiêu ngày để gửi yêu cầu trả hàng/hoàn tiền kể từ khi đơn hàng giao thành công?",
        "filter": None,
        "gold_doc": ["quy-dinh-chung-tra-hang-hoan-tien", "chinh-sach-tra-hang-hoan-tien"],
        "gold_snippet": "15 ngày",
        "gold_answer": "15 ngày kể từ lúc đơn hàng được cập nhật giao hàng thành công; riêng thực phẩm tươi sống và đông lạnh là 24 giờ.",
    },
    {
        "id": 2,
        "query": "Khi hệ thống ghi nhận đã trả hàng thành công nhưng Shop chưa nhận được hàng, Người Bán phải phản hồi Shopee trong bao lâu?",
        "filter": {"audience": "seller"},
        "gold_doc": ["quan-ly-don-tra-hang-hoan-tien-seller"],
        "gold_snippet": "2 ngày",
        "gold_answer": "Trong vòng 2 ngày, kể từ ngày hệ thống cập nhật trả hàng thành công.",
    },
    {
        "id": 3,
        "query": "Nếu người mua nhận hoàn tiền qua Ví ShopeePay, sau khi Shopee chấp nhận hoàn tiền thì mất bao lâu để nhận được tiền?",
        "filter": None,
        "gold_doc": ["thoi-gian-nhan-tien-hoan"],
        "gold_snippet": "24 giờ",
        "gold_answer": "24 giờ, với điều kiện Ví ShopeePay vẫn hoạt động bình thường.",
    },
    {
        "id": 4,
        "query": "Khi đóng gói hàng hoàn trả, người mua có được dán/viết thông tin trả hàng lên hộp của nhà sản xuất không?",
        "filter": None,
        "gold_doc": ["cach-dong-goi-hang-hoan-tra"],
        "gold_snippet": "hộp của nhà sản xuất",
        "gold_answer": "Không — không được dán/viết lên hộp của nhà sản xuất; cần dùng hộp vận chuyển bên ngoài để bảo vệ sản phẩm và hộp gốc.",
    },
    {
        "id": 5,
        "query": "Sau khi người mua gửi yêu cầu Trả hàng/Hoàn tiền, Shopee phản hồi kết quả xử lý trong khoảng thời gian nào?",
        "filter": None,
        "gold_doc": ["huong-dan-gui-yeu-cau-tra-hang"],
        "gold_snippet": "3 - 5 ngày",
        "gold_answer": "Khoảng 3-5 ngày làm việc.",
    },
]


class HeadingChunker:
    """Custom strategy: split on numbered/lettered heading lines (e.g. "1.1.", "A."),
    re-attaching the heading to every sub-piece when a section must be split further.
    Suited for policy documents authored as numbered clauses/sections (Vũ Minh Trí's role).
    """

    HEADING_RE = re.compile(r"^\s{0,3}(?:\d{1,2}(?:\.\d{1,2})*\.?\s+\S|[A-EĐ]\.\s+\S)")

    def __init__(self, chunk_size: int = 400) -> None:
        self.chunk_size = chunk_size
        self._fallback = RecursiveChunker(chunk_size=chunk_size)

    def chunk(self, text: str) -> list[str]:
        lines = text.split("\n")
        sections: list[tuple[str, str]] = []
        heading, buf = "", []
        for line in lines:
            if self.HEADING_RE.match(line.strip()) and line.strip():
                if buf:
                    sections.append((heading, "\n".join(buf).strip()))
                heading, buf = line.strip(), []
            else:
                buf.append(line)
        if buf:
            sections.append((heading, "\n".join(buf).strip()))

        chunks: list[str] = []
        for h, body in sections:
            full = f"{h}\n{body}".strip() if h else body
            if not full:
                continue
            if len(full) <= self.chunk_size:
                chunks.append(full)
            else:
                for piece in self._fallback.chunk(body):
                    combined = f"{h}\n{piece}".strip() if h else piece
                    if combined:
                        chunks.append(combined)
        return chunks


def load_corpus(data_dir: Path) -> list[tuple[dict, str, str]]:
    """Đọc từng file .md, tách frontmatter metadata và nội dung thân bài."""
    documents = []
    for md_file in sorted(data_dir.glob("*.md")):
        raw_text = md_file.read_text(encoding="utf-8")
        parts = raw_text.split("---")
        if len(parts) >= 3:
            fm_raw = parts[1].strip()
            body = "---".join(parts[2:]).strip()
            meta = dict(re.findall(r"^(\w+):\s*(.+)$", fm_raw, re.M))
            meta = {k: v.strip(" \"'") for k, v in meta.items()}
        else:
            meta = {}
            body = raw_text.strip()

        meta.setdefault("doc_id", md_file.stem)
        documents.append((meta, body, md_file.stem))
    return documents


def get_embedder():
    """Lấy embedding model cấu hình từ file .env"""
    load_dotenv(override=False)
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    if provider == "local":
        try:
            return LocalEmbedder(
                model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL)
            )
        except Exception:
            return _mock_embed
    elif provider == "openai":
        try:
            return OpenAIEmbedder(
                model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL)
            )
        except Exception:
            return _mock_embed
    elif provider == "gemini":
        try:
            return GeminiEmbedder(
                model_name=os.getenv("GEMINI_EMBEDDING_MODEL", GEMINI_EMBEDDING_MODEL)
            )
        except Exception as e:
            print(f"Lỗi khởi tạo GeminiEmbedder: {e}")
            return _mock_embed
    return _mock_embed


def run_benchmark(chunker=None, data_dir_str: str = "data/ecommerce") -> str:
    data_dir = Path(data_dir_str)
    if not data_dir.exists():
        return f"Error: Data directory '{data_dir}' not found."

    # Mặc định sử dụng RecursiveChunker nếu không chỉ định (có thể thay đổi sang SentenceChunker hoặc FixedSizeChunker)
    # chunk_size=400 để khớp với số liệu đã dẫn trong report/REPORT_CANHAN.md và REPORT_NHOM.md
    if chunker is None:
        chunker = RecursiveChunker(chunk_size=400)

    chunker_name = chunker.__class__.__name__
    raw_docs = load_corpus(data_dir)

    all_chunks: list[Document] = []
    for meta, body, file_stem in raw_docs:
        chunks = chunker.chunk(body)
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{file_stem}#{i}"
            chunk_meta = {**meta, "doc_id": file_stem, "chunk_index": i}
            all_chunks.append(
                Document(id=chunk_id, content=chunk_text, metadata=chunk_meta)
            )

    embedder = get_embedder()
    
    # Bọc lại embedder để tránh lỗi Rate Limit (100 request/phút) của Gemini và in ra tiến trình
    original_embedder_call = embedder.__call__
    def rate_limited_call(text):
        import time
        time.sleep(0.6) # Ngủ 0.6 giây để không vượt quá 100 req/phút
        return original_embedder_call(text)
    
    if "Gemini" in embedder.__class__.__name__:
        embedder.__call__ = rate_limited_call

    embedder_name = getattr(embedder, "_backend_name", embedder.__class__.__name__)

    store = EmbeddingStore(collection_name="benchmark_store", embedding_fn=embedder)
    
    print(f"Bắt đầu nạp {len(all_chunks)} chunks vào {embedder_name}...")
    print("Vui lòng chờ khoảng 1-2 phút (do API giới hạn số lần gọi/phút)...")
    for i, chunk in enumerate(all_chunks):
        if (i+1) % 10 == 0:
            print(f"  Đã nạp {i+1}/{len(all_chunks)} chunks...")
        store.add_documents([chunk])
    print("Đã nạp xong!")

    output_lines = []
    output_lines.append("=" * 70)
    output_lines.append(f"BENCHMARK RETRIEVAL REPORT — DAY 7 LAB")
    output_lines.append("=" * 70)
    output_lines.append(f"Chiến lược Chunking  : {chunker_name}")
    output_lines.append(f"Embedding Backend    : {embedder_name}")
    output_lines.append(f"Tổng số tài liệu gốc : {len(raw_docs)} files")
    output_lines.append(f"Tổng số chunks nạp   : {store.get_collection_size()} chunks")
    output_lines.append("-" * 70)

    naive_hits = 0
    real_hits = 0

    for q in BENCHMARK_QUERIES:
        q_id = q["id"]
        query = q["query"]
        q_filter = q["filter"]
        gold_docs = q["gold_doc"]
        gold_snippet = q["gold_snippet"]
        gold_ans = q["gold_answer"]

        output_lines.append(f"\n[QUERY {q_id}] {query}")
        if q_filter:
            output_lines.append(f"  * Metadata filter  : {q_filter}")
        output_lines.append(f"  * Gold Doc ID      : {gold_docs}")
        output_lines.append(f"  * Gold Answer      : {gold_ans}")

        results = store.search_with_filter(query, top_k=3, metadata_filter=q_filter)
        output_lines.append("  * Top-3 Retrieved Chunks:")

        found_doc = False
        for rank, res in enumerate(results, start=1):
            doc_id = res.get("metadata", {}).get("doc_id", "unknown")
            score = res.get("score", 0.0)
            preview = res["content"][:140].replace("\n", " ").strip()
            is_gold = doc_id in gold_docs
            if is_gold:
                found_doc = True
            tag = "[MATCH]" if is_gold else "       "
            output_lines.append(
                f"    {rank}. {tag} score={score:+.4f} | doc={doc_id} | id={res['id']}"
            )
            output_lines.append(f'       preview: "{preview}..."')

        # Chấm hai mức (theo docs/SCORING.md, tránh cách chấm ngây thơ chỉ dựa vào doc_id):
        # (a) top-3 có đúng file gold không; (b) ngữ cảnh top-3 có THỰC SỰ chứa chuỗi đáp án không.
        combined_context = " ".join(r["content"] for r in results)
        found_snippet = gold_snippet in combined_context
        naive_hits += found_doc
        real_hits += found_snippet

        output_lines.append(f"  => Đúng file gold trong top-3       : {'CO' if found_doc else 'KHONG'} (cach cham ngay tho)")
        output_lines.append(f"  => Ngu canh top-3 chua dap an ('{gold_snippet}') : {'CO' if found_snippet else 'KHONG'} (cach cham dung theo docs/SCORING.md)")

    output_lines.append("\n" + "=" * 70)
    output_lines.append(f"TONG KET: {naive_hits}/5 dung doc_id (cach cham ngay tho, de bi thoi phong)")
    output_lines.append(f"TONG KET: {real_hits}/5 THUC SU chua dap an trong ngu canh (cach cham dung)")

    report_text = "\n".join(output_lines)
    print(report_text)

    # Lưu kết quả ra file ket_qua_benchmark.txt
    Path("ket_qua_benchmark.txt").write_text(report_text, encoding="utf-8")
    return report_text


if __name__ == "__main__":
    # Mỗi thành viên đổi dòng dưới đây sang chiến lược của mình rồi chạy `python bench.py`:
    #   run_benchmark(chunker=RecursiveChunker(chunk_size=400))            # Mai Văn Trung
    run_benchmark(chunker=FixedSizeChunker(chunk_size=400, overlap=50))  # Ngô Văn Giáp
    #   run_benchmark(chunker=HeadingChunker(chunk_size=400))              # Vũ Minh Trí
    # run_benchmark() # Bị dư thừa, gây chạy 2 lần

