import streamlit as st
from core_logic import UltimateRAG
import os

st.set_page_config(layout="wide", page_title="Ultimate Financial RAG")

st.title("🚀 Ultimate Multimodal Financial Analyst")
st.markdown("---")

api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")

if api_key:
    rag = UltimateRAG(api_key)
    
    query = st.text_input("Ask a question about the Tesla 10-K (e.g., 'What are the risk factors regarding lithium?')")
    
    if query:
        with st.spinner("Analyzing with HyDE and Re-ranking..."):
            context_data = rag.retrieve_and_rerank(query) 
            
            final_text, source_data = rag.generate_final_answer(query, context_data)
            
            st.subheader("Final Analyst Report")
            st.write(final_text)
            
            st.sidebar.header("Sources & Evidence")
            
            for src in source_data:
                with st.sidebar.expander(f"Source: Page {src['page']} ({src['type'].upper()})"):
                    st.write(f"**Relevance Score:** `{src['score']:.2f}`")
                    
                    if src['type'] == 'visual':
                        st.info("Visual evidence used for analysis:")
                        img_path = os.path.join("Financial Analyst", src['file'])
                        if os.path.exists(img_path):
                            st.image(img_path)
                        else:
                            st.warning("Original image file not found.")
                    else:
                        st.markdown(src['text'])

else:
    st.warning("Please enter your Gemini API Key to start.")