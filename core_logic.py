import google.generativeai as genai
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
import numpy as np

class UltimateRAG:
    def __init__(self, gemini_api_key):
        genai.configure(api_key=gemini_api_key)

        self.llm = genai.GenerativeModel('gemini-2.5-flash-lite') 
        self.bi_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.re_ranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        self.client = chromadb.PersistentClient(path="Financial Analyst/chroma_db")
        self.collection = self.client.get_collection(name="tesla_reports")

    def generate_hyde_answer(self, query):
        prompt = f"""
        Act as a CFA (Chartered Financial Analyst). 
        Write a factual, data-heavy paragraph that answers the question: '{query}'
        Use technical terms like 'Adjusted EBITDA', 'Vesting Tranches', and 'GAAP metrics'.
        """
        response = self.llm.generate_content(prompt)
        return response.text

    def retrieve_and_rerank(self, query, top_k=10):
        hypothetical_answer = self.generate_hyde_answer(query)
        query_embedding = self.bi_encoder.encode([hypothetical_answer]).tolist()
        
        initial_results = self.collection.query(query_embeddings=query_embedding, n_results=20)
        initial_metas = initial_results['metadatas'][0]
        

        top_pages = list(set([m['page_number'] for m in initial_metas[:5]]))
        
        page_results = self.collection.get(
            where={"page_number": {"$in": top_pages}}
        )
        
        all_docs = page_results['documents']
        all_metas = page_results['metadatas']

        pairs = [[query, doc] for doc in all_docs]
        scores = self.re_ranker.predict(pairs)
        
        boosted_results = []
        for score, doc, meta in zip(scores, all_docs, all_metas):
            final_score = score
            if meta['content_type'] == 'table':
                final_score += 0.2 
            boosted_results.append((final_score, doc, meta))

        reranked_results = sorted(boosted_results, key=lambda x: x[0], reverse=True)
        return reranked_results[:top_k]

    def generate_final_answer(self, query, context_list):
        llm_context = ""
        ui_sources = []

        for score, text, meta in context_list:
            if meta['content_type'] == 'table':
                llm_context += f"\n[SOURCE: PAGE {meta['page_number']} - TABLE]:\n{text}\n"
            else:
                llm_context += f"\n[SOURCE: PAGE {meta['page_number']} - TEXT]:\n{text}\n"

            display_text = text.split("---UI_SEPARATOR---")[-1].strip() if "---UI_SEPARATOR---" in text else text
            ui_sources.append({
                "text": display_text,
                "score": float(score),
                "page": meta['page_number'],
                "type": meta['content_type'],
                "file": meta.get("file_name", "")
            })
        
        prompt = f"""
        You are a Senior Financial Analyst. Answer the user question based ONLY on the data below.
        
        DATA:
        {llm_context}
        
        QUESTION: {query}
        
        Provide a detailed, data-driven answer with relevant financial metrics and terminology.
        """
        response = self.llm.generate_content(prompt)
        return response.text, ui_sources

