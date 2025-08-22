# app.py (채팅 기록 저장/불러오기 기능 추가 최종본)

from PIL import Image # [신규] 로고 이미지를 처리하기 위해 Pillow 라이브러리 임포트
import nest_asyncio
nest_asyncio.apply()

import streamlit as st
import os
import shutil
import re
import json # [신규] 채팅 기록 저장을 위해 json 라이브러리 임포트
from datetime import datetime # [신규] 파일 이름에 현재 시간을 넣기 위해 임포트
from dotenv import load_dotenv

import rag_core
from config import DOCS_DIR, KNOWLEDGE_BASE_DIR


st.set_page_config(
    layout="wide",
    page_icon="assets/Project_logo.png" 
)
load_dotenv()
os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)

CREATE_NEW_KB_OPTION = "-- Create New Knowledge Base --"

# --- 세션 상태 초기화 ---
# ... (이전과 동일)
if 'api_provider' not in st.session_state: st.session_state['api_provider'] = 'NVIDIA'
if 'language' not in st.session_state: st.session_state['language'] = 'English'
if "messages" not in st.session_state: st.session_state.messages = []
if "retriever" not in st.session_state: st.session_state.retriever = None
if "api_key_ok" not in st.session_state: st.session_state.api_key_ok = False
if "selected_kb" not in st.session_state: st.session_state.selected_kb = None
if "user_api_key" not in st.session_state: st.session_state.user_api_key = None

# --- 언어별 텍스트 (채팅 기록 관련 텍스트 추가) ---
# --- 언어별 텍스트 (모든 수정 사항 최종 반영) ---
LANG_TEXT = {
    'English': {
        'page_title': "Chat with your AI Assistant, Envie!",
        'settings_header': "Settings",
        'api_select_label': "Select AI Provider",
        'lang_select_label': "Language",
        'kb_select_label': "Select Knowledge Base",
        'kb_reset_button': "Delete Selected Knowledge Base",
        'kb_reset_success': "Knowledge Base '{kb_name}' has been deleted.",
        'new_kb_header': "Create New Knowledge Base",
        'new_kb_name_label': "Enter a name for the new Knowledge Base:",
        'new_kb_name_help': "Only English letters, numbers, hyphens (-), and underscores (_) are allowed.",
        'invalid_kb_name_error': "Invalid name. Please use only English letters, numbers, hyphens (-), and underscores (_).",
        'upload_label': "Upload files for the new Knowledge Base:",
        'create_button': "Create!",
        'upload_success': "File {file_name} uploaded successfully!",
        'creating_db': "Creating Knowledge Base '{kb_name}'...",
        'db_created_success': "Knowledge Base '{kb_name}' created.",
        'chat_placeholder': "Ask me anything about the documents!",
        'system_prompt': "You are a helpful AI assistant named Envie. If provided with context, use it to inform your responses. If no context is available, use your general knowledge to provide a helpful response.",
        'update_kb_header': "Update Selected Knowledge Base",
        'update_upload_label': "Upload additional files:",
        'update_button': "Add to Knowledge Base",
        'updating_db': "Adding files to '{kb_name}'...",
        'db_updated_success': "Knowledge Base '{kb_name}' updated successfully.",
        'api_key_header': "Enter Your API Key",
        'api_key_label': "Your {api_provider} API Key (Optional)",
        'api_key_help': "If you do not provide a key, the key from the .env file will be used (for local testing).",
        'api_key_missing_error': "Please enter your API key in the sidebar, or set it in the .env file to activate the AI.",
        'chat_history_header': "Chat History",
        'chat_history_save_button': "Save Chat",
        'chat_history_load_label': "Load Chat",
        'api_key_source_label': "API Key Source",
        'api_key_source_local': "Use Local (.env file)",
        'api_key_source_user': "Enter Manually",

    },
    'Korean': {
        'page_title': "AI 어시스턴트, Envie와 대화하기",
        'settings_header': "설정",
        'api_select_label': "AI 모델 선택",
        'lang_select_label': "언어",
        'kb_select_label': "지식 베이스 선택",
        'kb_reset_button': "선택한 지식 베이스 삭제",
        'kb_reset_success': "'{kb_name}' 지식 베이스가 삭제되었습니다.",
        'new_kb_header': "새로운 지식 베이스 만들기",
        'new_kb_name_label': "새 지식 베이스의 이름을 입력하세요:",
        'new_kb_name_help': "이름은 영문, 숫자, 하이픈(-), 언더스코어(_)만 사용할 수 있습니다.",
        'invalid_kb_name_error': "이름이 유효하지 않습니다. 영문, 숫자, 하이픈(-), 언더스코어(_)만 사용해주세요.",
        'upload_label': "새 지식 베이스에 사용할 파일을 업로드하세요:",
        'create_button': "생성하기!",
        'upload_success': "파일 {file_name} 업로드 성공!",
        'creating_db': "'{kb_name}' 지식 베이스를 생성하는 중...",
        'db_created_success': "'{kb_name}' 지식 베이스가 생성되었습니다.",
        'chat_placeholder': "문서에 대해 무엇이든 물어보세요!",
        'system_prompt': "당신은 Envie라는 이름의 도움이 되는 AI 어시스턴트입니다. 컨텍스트가 제공되면 응답에 참고하세요. 컨텍스트가 없으면 일반 지식을 사용하여 유용한 답변을 제공하세요. 모든 답변은 반드시 한국어로 작성해야 합니다.",
        'update_kb_header': "선택한 지식 베이스 업데이트",
        'update_upload_label': "추가할 파일을 업로드하세요:",
        'update_button': "지식 베이스에 추가",
        'updating_db': "'{kb_name}'에 파일을 추가하는 중...",
        'db_updated_success': "'{kb_name}' 지식 베이스가 성공적으로 업데이트되었습니다.",
        'api_key_header': "API 키 입력",
        'api_key_label': "{api_provider} API 키 (선택 사항)",
        'api_key_help': "키를 입력하지 않으면 .env 파일의 키를 사용합니다 (로컬 테스트용).",
        'api_key_missing_error': "AI를 활성화하려면 사이드바에 API 키를 입력하거나 .env 파일에 키를 설정해주세요.",
        'chat_history_header': "대화 기록",
        'chat_history_save_button': "대화 내용 저장",
        'chat_history_load_label': "대화 내용 불러오기",
        'api_key_source_label': "API 키 사용 방식",
        'api_key_source_local': "로컬 (.env 파일)",
        'api_key_source_user': "직접 입력",

    }
}
lang = LANG_TEXT[st.session_state['language']]

# ... (헬퍼 함수 및 모델 로딩 함수는 이전과 동일)
def get_knowledge_bases(): return [d for d in os.listdir(KNOWLEDGE_BASE_DIR) if os.path.isdir(os.path.join(KNOWLEDGE_BASE_DIR, d))]
def is_valid_kb_name(name): return re.match("^[A-Za-z0-9_-]+$", name) is not None
def load_llm_and_embedder(api_provider, user_api_key): return rag_core.load_models(api_provider, user_api_key, st.error)

# --- 사이드바 UI ---
with st.sidebar:
    # [수정] try...except 블록의 문법과 들여쓰기를 다시 한번 확인합니다.
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        logo_path = os.path.join(script_dir, "assets", "Project_logo.png")
        logo_image = Image.open(logo_path)
        st.image(logo_image)
    except FileNotFoundError:
        pass
    # ... (이전 사이드바 코드는 동일)
    st.subheader(lang['settings_header'])
    selected_api = st.selectbox(lang['api_select_label'], ['NVIDIA', 'Google'], index=0 if st.session_state.api_provider == 'NVIDIA' else 1)
    if selected_api != st.session_state.api_provider:
        st.session_state.api_provider = selected_api; 
        st.session_state.messages = []; st.session_state.retriever = None
        st.session_state.api_key_ok = False
        st.session_state.user_api_key = None; st.rerun()
    selected_language = st.selectbox(lang['lang_select_label'], ['English', 'Korean'], index=0 if st.session_state.language == 'English' else 1)
    if selected_language != st.session_state.language:
        st.session_state.language = selected_language; st.session_state.messages = []; st.rerun()

    st.divider()
    
    # [신규] API 키 사용 방식을 선택하는 라디오 버튼
    st.subheader(lang['api_key_header'])
    api_key_source = st.radio(
        label=lang['api_key_source_label'],
        options=[lang['api_key_source_local'], lang['api_key_source_user']],
        key="api_key_source_widget"
    )

    # "직접 입력"을 선택했을 때만 API 키 입력창을 보여줌
    if api_key_source == lang['api_key_source_user']:
        user_api_key_input = st.text_input(
            label=lang['api_key_label'].format(api_provider=st.session_state.api_provider),
            type="password",
            help=lang['api_key_help'],
            key="api_key_input_widget"
        )
        if user_api_key_input and user_api_key_input != st.session_state.user_api_key:
            st.session_state.user_api_key = user_api_key_input
            st.session_state.api_key_ok = False
            st.rerun()
    else:
        # ".env 파일"을 선택하면, 사용자 입력 키는 None으로 설정
        st.session_state.user_api_key = None
        
    st.divider()
    # --- 모델 로딩 및 키 유효성 검사 ---
    # 사이드바에서 사용자가 선택한 "API 키 사용 방식"에 따라 어떤 키를 사용할지 결정합니다.
    final_key_to_use = None
    if api_key_source == lang['api_key_source_local']:
        # 옵션 1: "로컬 (.env 파일)"을 선택한 경우
        # .env 파일에서 해당 AI 제공사의 API 키를 가져옵니다.
        final_key_to_use = os.getenv(f"{st.session_state.api_provider.upper()}_API_KEY")
    else: 
        # 옵션 2: "직접 입력"을 선택한 경우
        # 사용자가 입력하여 세션 상태에 저장된 API 키를 가져옵니다.
        final_key_to_use = st.session_state.user_api_key

    # 최종적으로 결정된 API 키를 사용하여 모델 로딩 함수를 호출합니다.
    llm, embedder = load_llm_and_embedder(st.session_state.api_provider, final_key_to_use)

    # 모델과 임베더가 성공적으로 로드되었는지 (None이 아닌지) 확인하여
    # api_key_ok 상태를 업데이트합니다. 이 상태는 앱의 나머지 부분에서 AI 기능 활성화 여부를 결정합니다.
    st.session_state.api_key_ok = llm is not None and embedder is not None
    

    if st.session_state.api_key_ok:
        kb_list = get_knowledge_bases()
        kb_options = [CREATE_NEW_KB_OPTION] + kb_list
        selected_kb = st.selectbox(lang['kb_select_label'], options=kb_options, key="selected_kb_widget")
        if st.session_state.selected_kb != selected_kb:
            st.session_state.selected_kb = selected_kb; st.session_state.retriever = None; st.rerun()
        if st.session_state.selected_kb == CREATE_NEW_KB_OPTION:
            st.subheader(lang['new_kb_header'])
            with st.form("new_kb_form"):
                new_kb_name = st.text_input(lang['new_kb_name_label'], help=lang['new_kb_name_help'])
                uploaded_files = st.file_uploader(lang['upload_label'], accept_multiple_files=True)
                submitted = st.form_submit_button(lang['create_button'])
                if submitted:
                    if not new_kb_name or not is_valid_kb_name(new_kb_name): st.error(lang['invalid_kb_name_error'])
                    elif not uploaded_files: st.warning("Please upload at least one file.")
                    else:
                        if os.path.exists(DOCS_DIR): shutil.rmtree(DOCS_DIR)
                        os.makedirs(DOCS_DIR)
                        for file in uploaded_files:
                            with open(os.path.join(DOCS_DIR, file.name), "wb") as f: f.write(file.read())
                        with st.spinner(lang['creating_db'].format(kb_name=new_kb_name)):
                            rag_core.create_and_save_retriever(embedder, new_kb_name)
                            st.success(lang['db_created_success'].format(kb_name=new_kb_name))
                            st.session_state.selected_kb = new_kb_name; st.rerun()
        elif st.session_state.selected_kb and st.session_state.selected_kb != CREATE_NEW_KB_OPTION:
            st.divider()
            st.subheader(lang['update_kb_header'])
            with st.form("update_kb_form"):
                update_files = st.file_uploader(lang['update_upload_label'], accept_multiple_files=True)
                update_submitted = st.form_submit_button(lang['update_button'])
                if update_submitted and update_files:
                    if os.path.exists(DOCS_DIR): shutil.rmtree(DOCS_DIR)
                    os.makedirs(DOCS_DIR)
                    for file in update_files:
                        with open(os.path.join(DOCS_DIR, file.name), "wb") as f: f.write(file.read())
                    with st.spinner(lang['updating_db'].format(kb_name=st.session_state.selected_kb)):
                        st.session_state.retriever = rag_core.update_and_save_retriever(embedder, st.session_state.selected_kb)
                        st.success(lang['db_updated_success'].format(kb_name=st.session_state.selected_kb))
                    if os.path.exists(DOCS_DIR): shutil.rmtree(DOCS_DIR)
            st.divider()
            if st.button(lang['kb_reset_button']):
                kb_to_delete = st.session_state.selected_kb; kb_path = os.path.join(KNOWLEDGE_BASE_DIR, kb_to_delete)
                if os.path.exists(kb_path): shutil.rmtree(kb_path)
                st.session_state.retriever = None; st.session_state.selected_kb = None
                st.success(lang['kb_reset_success'].format(kb_name=kb_to_delete)); st.rerun()
    
    # [신규] 채팅 기록 저장 및 불러오기 UI
    st.divider()
    st.subheader(lang['chat_history_header'])
    
    # 1. 저장 기능
    # 현재 시간을 포함하여 고유한 파일 이름 생성
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"chat_history_{now}.json"
    # st.download_button을 위한 데이터 준비 (json 형식의 문자열)
    chat_history_json = json.dumps(st.session_state.messages, indent=2, ensure_ascii=False)
    
    st.download_button(
        label=lang['chat_history_save_button'],
        data=chat_history_json,
        file_name=file_name,
        mime="application/json"
    )

    # 2. 불러오기 기능
    loaded_chat_file = st.file_uploader(
        label=lang['chat_history_load_label'],
        type=['json']
    )
    if loaded_chat_file is not None:
        try:
            # 업로드된 파일의 내용을 읽고 json으로 파싱
            loaded_messages = json.load(loaded_chat_file)
            # st.session_state.messages를 불러온 내용으로 교체
            st.session_state.messages = loaded_messages
            # 화면을 새로고침하여 불러온 대화 내용 표시
            st.rerun()
        except json.JSONDecodeError:
            st.error("잘못된 JSON 파일 형식입니다. 올바른 채팅 기록 파일을 업로드해주세요.")
        except Exception as e:
            st.error(f"파일을 불러오는 중 오류가 발생했습니다: {e}")


# --- 리트리버 준비 ---
# ... (이하 코드는 이전과 동일)
if st.session_state.api_key_ok and st.session_state.retriever is None:
    if st.session_state.selected_kb and st.session_state.selected_kb != CREATE_NEW_KB_OPTION:
        with st.spinner(f"Loading '{st.session_state.selected_kb}'..."):
            st.session_state.retriever = rag_core.load_retriever(embedder, st.session_state.selected_kb)
        if st.session_state.retriever: st.sidebar.success(f"'{st.session_state.selected_kb}' loaded.")

# --- 채팅 인터페이스 ---
final_page_title = lang['page_title']
if st.session_state.get('api_provider') == 'NVIDIA': final_page_title += " with NVIDIA"
elif st.session_state.get('api_provider') == 'Google': final_page_title += " with Google"
st.subheader(final_page_title)

for message in st.session_state.messages:
    with st.chat_message(message["role"]): st.markdown(message["content"])

if not st.session_state.api_key_ok:
    st.info(lang['api_key_missing_error'])
elif not st.session_state.retriever:
    st.info("Please select a Knowledge Base from the sidebar, or create a new one.")
else:
    rag_chain = rag_core.create_rag_chain(llm, st.session_state.retriever, lang['system_prompt'])
    user_input = st.chat_input(lang['chat_placeholder'])
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"): st.markdown(user_input)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            responses = rag_core.get_contextual_response(user_input, st.session_state.retriever, rag_chain)
            for response in responses:
                full_response += response; message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})