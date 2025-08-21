import streamlit as st

st.title("메인 화면 컴포넌트 예시")

# 1. 컬럼으로 나누기
col1, col2 = st.columns(2)

with col1:
    st.header("첫 번째 열")
    st.button("왼쪽 버튼")

with col2:
    st.header("두 번째 열")
    st.slider("오른쪽 슬라이더", 0, 100, 50)

st.divider() # 구분선 추가

# 2. 탭으로 나누기
tab1, tab2 = st.tabs(["입력", "결과"])

with tab1:
    st.header("입력 탭")
    user_input = st.text_input("여기에 텍스트를 입력하세요.")
    st.write(f"입력된 텍스트: {user_input}")

with tab2:
    st.header("결과 탭")
    if user_input:
        st.success("입력된 내용이 성공적으로 처리되었습니다!")