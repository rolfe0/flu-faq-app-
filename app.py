import streamlit as st
import json
import os
import io
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
from dotenv import load_dotenv
from docx import Document

# 關閉 jieba 啟動時的繁瑣訊息
jieba.setLogLevel(20)

# 讀取環境變數
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# 初始化 Gemini API
has_api_key = bool(API_KEY)
client = genai.Client(api_key=API_KEY) if has_api_key else None


# 1. R（Retrieval 檢索模組）：jieba 斷詞 + TF-IDF + 餘弦相似度
class FluRAGRetriever:
    def __init__(self, faq_list: list):
        self.faq_list = faq_list
        self.vectorizer = TfidfVectorizer()
        self.doc_vectors = None
        if self.faq_list:
            self._prepare()

    def _prepare(self):
        corpus = [
            f"{item['question']} {item['source_text']}"
            for item in self.faq_list
        ]
        docs_cut = [" ".join(jieba.cut(doc)) for doc in corpus]
        self.doc_vectors = self.vectorizer.fit_transform(docs_cut)

    def retrieve(self, user_query: str, top_k: int = 2, threshold: float = 0.05):
        if not self.faq_list or self.doc_vectors is None:
            return []

        query_cut = " ".join(jieba.cut(user_query))
        query_vec = self.vectorizer.transform([query_cut])
        similarities = cosine_similarity(query_vec, self.doc_vectors)[0]

        ranked = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in ranked[:top_k]:
            if score >= threshold:
                item_with_score = dict(self.faq_list[idx])
                item_with_score["similarity_score"] = round(float(score), 3)
                results.append(item_with_score)
        return results


@st.cache_resource
def init_rag_system():
    try:
        with open("data/flu_faq.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        retriever = FluRAGRetriever(data)
        return data, retriever
    except FileNotFoundError:
        st.error("找不到衛教資料檔 (data/flu_faq.json)，請確認檔案結構。")
        return [], None


# 2. G（Generation 生成模組）：結合檢索文獻與安全規範
def generate_rag_answer(user_question: str, retrieved_docs: list):
    context_blocks = []
    for i, doc in enumerate(retrieved_docs, start=1):
        context_blocks.append(
            f"[參考資料 {i}]（標題：{doc['source_title']}）\n{doc['source_text']}"
        )
    combined_context = "\n\n".join(context_blocks)

    prompt = f"""你是一般醫療衛教文字整理助手。請「嚴格只依據」下方檢索到的「參考資料」回答「使用者問題」。
要求：
1. 請用繁體中文，以條列式、簡短且一般民眾容易理解的白話文撰寫。
2. 嚴禁補充參考資料中沒有的醫療資訊，嚴禁進行疾病診斷、開藥或提供個人化治療建議。
3. 如果參考資料的內容不足以回答使用者的問題，請明確回答：「目前衛教資料庫的資料不足以完整回答此問題，建議您查看下方官方來源或諮詢專業醫療人員。」

使用者問題：{user_question}

檢索到的參考資料：
{combined_context}"""

    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt
        )
        return response.text or "API_ERROR: 模型未回傳文字"
    except Exception as e:
        return f"API_ERROR: {str(e)}"


# 3. Word 衛教報告匯出模組
def create_word_report(question: str, answer: str, retrieved_docs: list) -> bytes:
    doc = Document()
    doc.add_heading("流感預防 RAG 智慧衛教諮詢報告", level=0)

    doc.add_paragraph(
        "⚠️ 免責聲明：本報告內容由 AI 依據衛生福利部疾病管制署（CDC）公開衛教資料整理，"
        "僅供一般衛生教育參考，不能取代專業醫師診斷或個人化醫療建議。"
    )

    doc.add_heading("一、 使用者諮詢問題", level=1)
    doc.add_paragraph(question)

    doc.add_heading("二、 AI 衛教整理回答", level=1)
    doc.add_paragraph(answer)

    doc.add_heading("三、 RAG 檢索參考文獻與來源追溯", level=1)
    for i, item in enumerate(retrieved_docs, start=1):
        doc.add_paragraph(
            f"文獻 {i}：{item['source_title']}（檢索相似度：{item['similarity_score']}）",
            style="List Bullet"
        )
        doc.add_paragraph(f"官方連結：{item['source_url']}")
        doc.add_paragraph(f"資料核對日期：{item['checked_date']}")

    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer.getvalue()


# 4. 網站前端介面
st.set_page_config(
    page_title="流感預防 RAG 智慧衛教系統",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ 流感預防 RAG 智慧衛教系統")
st.caption("結合 TF-IDF 語意檢索（Retrieval）與大語言模型生成（Generation）之衛教輔助原型")

st.warning(
    "⚠️ **重要提醒**：本系統只提供一般衛教資訊，不能取代醫療人員判斷。"
    "本系統不提供疾病診斷、個人化治療或用藥建議。"
)

faq_data, retriever = init_rag_system()

if faq_data and retriever:
    with st.expander("💡 不知道怎麼問？點此查看範例提問"):
        st.markdown(
            "- 平常生活中要怎麼預防流感？\n"
            "- 哪些人可以打公費流感疫苗？\n"
            "- 流感跟一般感冒有什麼不一樣？\n"
            "- 出現哪些危險徵兆要趕快去大醫院？"
        )

    user_query = st.text_input(
        "請輸入您想了解的流感衛教問題：",
        placeholder="例如：平常可以如何預防流感？或接種疫苗要注意什麼？"
    )

    if st.button("🔍 檢索並產生回答", type="primary"):
        if not user_query.strip():
            st.warning("請先輸入您的問題再點擊按鈕。")
        else:
            with st.spinner("正在從疾管署知識庫檢索最相關文獻..."):
                retrieved_docs = retriever.retrieve(user_query, top_k=2, threshold=0.05)

            if not retrieved_docs:
                st.error(
                    "🔍 **檢索結果**：目前衛教資料庫中找不到與您問題高度相關的流感資料。\n\n"
                    "請嘗試輸入與「流感預防、疫苗接種、症狀照護」相關的關鍵字，或撥打免付費防疫專線 **1922** 洽詢。"
                )
            else:
                with st.spinner("AI 正在根據檢索到的官方資料整理白話回答..."):
                    if has_api_key and client:
                        answer = generate_rag_answer(user_query, retrieved_docs)

                        if answer.startswith("API_ERROR"):
                            st.error("API 呼叫失敗，已自動切換為離線示範模式。")
                            st.code(answer)
                            final_answer = retrieved_docs[0]["source_text"]
                            st.info("以下為檢索到的最相關官方原始資料：")
                            st.write(final_answer)
                        else:
                            final_answer = answer
                            st.success("✅ RAG 衛教回答已生成：")
                            st.write(final_answer)
                    else:
                        st.info("系統未偵測到 API 金鑰，目前為離線示範模式（僅顯示檢索結果）。")
                        final_answer = retrieved_docs[0]["source_text"]
                        st.write(f"**最相關官方資料**：{final_answer}")

                st.markdown("---")

                st.markdown("### 📚 RAG 檢索依據與資料來源")
                for i, doc in enumerate(retrieved_docs, start=1):
                    with st.container(border=True):
                        st.markdown(
                            f"**參考文獻 {i}：{doc['source_title']}** "
                            f"（餘弦相似度：`{doc['similarity_score']}`）"
                        )
                        st.write(f"- **對應題庫主題**：{doc['question']}")
                        st.write(f"- **官方連結**：[{doc['source_url']}]({doc['source_url']})")
                        st.write(f"- **資料核對日期**：{doc['checked_date']}")

                docx_bytes = create_word_report(user_query, final_answer, retrieved_docs)
                st.download_button(
                    label="📥 下載本次衛教諮詢報告 (.docx)",
                    data=docx_bytes,
                    file_name="流感預防衛教諮詢報告.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )