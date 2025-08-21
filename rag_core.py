# rag_core.py (PDF 로더 기능 추가 최종본)

import os
import pickle
from dotenv import load_dotenv

from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# [신규] 다양한 파일 로더를 임포트합니다.
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from config import (
    DOCS_DIR, KNOWLEDGE_BASE_DIR,
    PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP
)

load_dotenv()

# [신규] DOCS_DIR에 있는 모든 문서를 확장자에 맞게 로드하는 함수
def load_documents_from_directory(directory):
    """디렉토리 내의 모든 문서를 확장자에 맞는 로더를 사용하여 로드합니다."""
    all_documents = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            try:
                if filename.lower().endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                    print(f"PDF 로더로 '{filename}' 처리 중...")
                elif filename.lower().endswith(".txt"):
                    loader = TextLoader(file_path, encoding='utf-8')
                    print(f"TXT 로더로 '{filename}' 처리 중...")
                else:
                    # 지원하지 않는 다른 파일 형식은 건너뜁니다.
                    print(f"지원하지 않는 파일 형식: '{filename}'")
                    continue
                
                # 로더를 사용하여 문서를 로드하고 리스트에 추가합니다.
                all_documents.extend(loader.load())
            except Exception as e:
                print(f"'{filename}' 파일 처리 중 오류 발생: {e}")

    return all_documents


# --- (이하 load_models, get_splitters 함수는 이전과 동일) ---
def load_models(api_provider, api_key, st_error_func):
    """주어진 API 키를 사용하여 LLM과 임베더를 로드합니다."""

    # 이제 이 함수는 키가 어디서 왔는지 고민할 필요가 없습니다.
    # 키가 없으면 그냥 없다고 알려주기만 하면 됩니다.
    if not api_key:
        st_error_func(f"{api_provider} API 키가 제공되지 않았습니다.")
        return None, None
        
    try:
        if api_provider == 'NVIDIA':
            llm = ChatNVIDIA(model="mistralai/mixtral-8x7b-instruct-v0.1", nvidia_api_key=api_key)
            embedder = NVIDIAEmbeddings(model="nvidia/nvolve-embed-v1", nvidia_api_key=api_key)
        elif api_provider == 'Google':
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
            embedder = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        else:
            return None, None
        return llm, embedder
    except Exception as e:
        st_error_func(f"{api_provider} 모델 로딩 중 오류 발생: 잘못된 API 키일 수 있습니다. ({e})")
        return None, None
    
def get_splitters():
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=PARENT_CHUNK_SIZE, chunk_overlap=PARENT_CHUNK_OVERLAP)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=CHILD_CHUNK_SIZE, chunk_overlap=CHILD_CHUNK_OVERLAP)
    return parent_splitter, child_splitter

# [변경] DirectoryLoader 대신 우리가 만든 새로운 로더 함수를 사용하도록 수정
def create_and_save_retriever(embedder, kb_name):
    if not os.path.exists(DOCS_DIR) or not os.listdir(DOCS_DIR): return None
    
    # DirectoryLoader 대신 새로운 함수를 호출합니다.
    raw_documents = load_documents_from_directory(DOCS_DIR)

    if not raw_documents:
        print("로드할 수 있는 문서가 없습니다.") 
        return None

    parent_splitter, child_splitter = get_splitters()
    vectorstore = FAISS.from_documents(raw_documents, embedder)
    store = InMemoryStore()
    retriever = ParentDocumentRetriever(vectorstore=vectorstore, docstore=store, child_splitter=child_splitter, parent_splitter=parent_splitter)
    retriever.add_documents(raw_documents, ids=None)
    kb_path = os.path.join(KNOWLEDGE_BASE_DIR, kb_name)
    os.makedirs(kb_path, exist_ok=True)
    retriever.vectorstore.save_local(os.path.join(kb_path, "faiss_index"))
    with open(os.path.join(kb_path, "docstore.pkl"), "wb") as f: pickle.dump(retriever.docstore, f)
    return retriever

def load_retriever(embedder, kb_name):
    try:
        kb_path = os.path.join(KNOWLEDGE_BASE_DIR, kb_name)
        vectorstore = FAISS.load_local(os.path.join(kb_path, "faiss_index"), embedder, allow_dangerous_deserialization=True)
        with open(os.path.join(kb_path, "docstore.pkl"), "rb") as f: store = pickle.load(f)
        parent_splitter, child_splitter = get_splitters()
        return ParentDocumentRetriever(vectorstore=vectorstore, docstore=store, child_splitter=child_splitter, parent_splitter=parent_splitter)
    except Exception as e:
        print(f"리트리버 '{kb_name}' 로딩 실패: {e}"); return None

# [변경] DirectoryLoader 대신 우리가 만든 새로운 로더 함수를 사용하도록 수정
def update_and_save_retriever(embedder, kb_name):
    retriever = load_retriever(embedder, kb_name)
    if not retriever: return None
    if not os.path.exists(DOCS_DIR) or not os.listdir(DOCS_DIR): return retriever
    
    # DirectoryLoader 대신 새로운 함수를 호출합니다.
    new_documents = load_documents_from_directory(DOCS_DIR)

    if not new_documents: return retriever
    retriever.add_documents(new_documents, ids=None)
    kb_path = os.path.join(KNOWLEDGE_BASE_DIR, kb_name)
    retriever.vectorstore.save_local(os.path.join(kb_path, "faiss_index"))
    with open(os.path.join(kb_path, "docstore.pkl"), "wb") as f: pickle.dump(retriever.docstore, f)
    return retriever

# --- (이하 RAG 체인 관련 함수는 변경 없음) ---
def create_rag_chain(llm, retriever, system_prompt):
    prompt_template = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
    chain = prompt_template | llm | StrOutputParser()
    return chain

def get_contextual_response(user_input, retriever, chain):
    docs = retriever.invoke(user_input)
    context = "\n\n".join([doc.page_content for doc in docs])
    augmented_user_input = f"Context: {context}\n\nQuestion: {user_input}\n"
    return chain.stream({"input": augmented_user_input})