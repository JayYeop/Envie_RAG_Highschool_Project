# config.py

import os

# --- 1. Directory Settings ---
# 문서가 업로드될 디렉토리
DOCS_DIR = os.path.abspath("./uploaded_docs")
# [변경] 모든 지식 베이스(KB)를 저장할 최상위 '도서관' 폴더
KNOWLEDGE_BASE_DIR = os.path.abspath("./knowledge_bases")

# --- 2. ParentDocumentRetriever Settings ---
# 컨텍스트 제공에 사용될 '부모 청크'의 크기
PARENT_CHUNK_SIZE = 1500
PARENT_CHUNK_OVERLAP = 200

# 유사도 검색에 사용될 '자식 청크'의 크기
# PARENT_CHUNK_SIZE 보다 작아야 합니다.
CHILD_CHUNK_SIZE = 400
CHILD_CHUNK_OVERLAP = 50