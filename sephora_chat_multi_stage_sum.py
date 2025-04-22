import streamlit as st
from openai import OpenAI
from pinecone import Pinecone
import tempfile
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
    with st.spinner("Ищу все релевантные отзывы..."):
        try:
            embedding = client.embeddings.create(
                input=query,
                model="text-embedding-3-small"
            ).data[0].embedding

            top_k = 1000  # или любое другое число, не превышающее лимит

            result = index.query(
                vector=embedding,
                top_k=top_k,
                include_metadata=True
            )

            rows = []
            for match in result["matches"]:
                meta = match.get("metadata", {})
                rows.append({
                    "id": match.get("id", ""),
                    "score": match.get("score", ""),
                    "brand": meta.get("brand", ""),
                    "product_name": meta.get("product_name", ""),
                    "review_text": meta.get("review_text", ""),
                    "rating": meta.get("rating", "")
                })

            if rows:
                df_results = pd.DataFrame(rows)
                tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name
                df_results.to_csv(tmp_path, index=False)
                st.success("✅ Отзывы собраны.")
                st.download_button("📥 Скачать отзывы", tmp_path, file_name="relevant_reviews.csv")
                st.dataframe(df_results.head(10))
            else:
                st.warning("🙁 Не удалось найти релевантные отзывы.")

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
