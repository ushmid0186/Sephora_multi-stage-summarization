import streamlit as st
from openai import OpenAI
from pinecone import Pinecone
import io
import pandas as pd


# ✅ получаем ключи из secrets
OPENAI_API_KEY   = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
INDEX_NAME       = st.secrets["INDEX_NAME"]

# ✅ инициализация
client = OpenAI(api_key=OPENAI_API_KEY)
pc     = Pinecone(api_key=PINECONE_API_KEY)
index  = pc.Index(INDEX_NAME)


# -------------------------
# 5. UI Styling and initialize history
# -------------------------
st.markdown("""
<style>
html, body, [class*="css"] { font-size:12px; }
.review-box { font-size:0.8em; border-bottom:1px dashed #ccc; margin-bottom:6px; padding-bottom:4px; }
.chat-history { max-height:400px; overflow-y:auto; border:1px solid #ccc; padding:6px; background-color:#f9f9f9; }
.stTextArea textarea { margin-bottom:0; font-size:12px !important; }
.submit-button-container button { margin-top:2px; width:100%; }
</style>
""", unsafe_allow_html=True)

if 'history' not in st.session_state:
    st.session_state.history = []

# -------------------------
# 6. Query input form
# -------------------------
st.title("🧴 Sephora Review Chat with Multi_Stage_Summary")
with st.form(key='query_form'):
    query = st.text_area('', placeholder='Type your question...', height=70, label_visibility='collapsed')
    submitted = st.form_submit_button('🔍')


if submitted and query:
    try:
        with st.spinner("🔎 Ищу все релевантные отзывы..."):
            # 1. Создание эмбеддинга для запроса
            embedding = client.embeddings.create(
                input=query,
                model="text-embedding-3-small"
            ).data[0].embedding

            # 2. Запрос в Pinecone (берем максимум 1000 отзывов)
            result = index.query(
                vector=embedding,
                top_k=1000,
                include_metadata=True
            )

            # 3. Сбор результатов
            rows = []
            for match in result["matches"]:
                meta = match["metadata"]
                rows.append({
                    "id": match["id"],
                    "score": match["score"],
                    "brand": meta.get("brand", ""),
                    "product_name": meta.get("product_name", ""),
                    "review_text": meta.get("review_text", ""),
                    "rating": meta.get("rating", "")
                })

            df_results = pd.DataFrame(rows)

            # 4. Превью в Streamlit
            st.success(f"✅ Найдено {len(df_results)} релевантных отзывов")
            st.dataframe(df_results.head(10))

            # 5. Сохраняем в CSV и даем скачать
            csv_buffer = io.StringIO()
            df_results.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()

            st.download_button(
                label="📥 Скачать все отзывы (CSV)",
                data=csv_data,
                file_name="relevant_reviews.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"❌ Ошибка при поиске отзывов: {e}")

# -------------------------
# 8. Render chat history & reviews
# -------------------------
left, right = st.columns([2, 1])
with left:
    st.markdown('### Chat history')
    st.markdown('<div class="chat-history">', unsafe_allow_html=True)
    for entry in reversed(st.session_state.history):
        st.markdown(f"**Q:** {entry['question']}")
        st.markdown(f"**A:** {entry['answer']}")
    st.markdown('</div>', unsafe_allow_html=True)
with right:
    st.markdown('### 📝 Reviews & clusters')
    if st.session_state.history:
        last = st.session_state.history[-1]
        st.markdown(f"**Total reviews used:** {len(last['reviews'])}")
        st.markdown('**Cluster distribution:**')
        for line in last['cluster_overview']:
            st.markdown(f"- {line}")
        for txt in last['reviews']:
            st.markdown(f"<div class='review-box'>{txt}</div>", unsafe_allow_html=True)
