import json
import google.generativeai as genai
from core_logic import UltimateRAG
import pandas as pd
import time

API_KEY = "---------------------------------------"
rag_system = UltimateRAG(API_KEY)
genai.configure(api_key=API_KEY)
judge_llm = genai.GenerativeModel('gemini-2.5-pro')

def get_judge_score(criterion, question, context, answer, ground_truth):
    prompt = f"""
    You are an impartial judge evaluating a RAG system.
    
    CRITERION: {criterion}
    QUESTION: {question}
    CONTEXT PROVIDED TO RAG: {context}
    RAG'S FINAL ANSWER: {answer}
    GROUND TRUTH ANSWER: {ground_truth}
    
    SCORING RUBRIC:
    - Faithfulness: Is the answer derived ONLY from the context? (0.0 to 1.0)
    - Relevance: Does the answer directly address the question? (0.0 to 1.0)
    
    Return ONLY a JSON object like this: {{"score": 0.9, "reason": "Brief explanation"}}
    """
    response = judge_llm.generate_content(prompt)
    try:
        json_str = response.text.strip().replace('```json', '').replace('```', '')
        return json.loads(json_str)
    except:
        return {"score": 0, "reason": "Error parsing judge response"}

with open("eval_dataset.json", "r") as f:
    test_set = json.load(f)

results = []

print(f" Starting Evaluation on {len(test_set)} questions...")

for item in test_set:
    print(f"Testing Question: {item['question'][:50]}...")
    
    context_chunks = rag_system.retrieve_and_rerank(item['question'])
    rag_answer, _ = rag_system.generate_final_answer(item['question'], context_chunks)

    time.sleep(6)
    
    context_text = "\n".join([c[1] for c in context_chunks])
    
    faithfulness = get_judge_score("Faithfulness", item['question'], context_text, rag_answer, item['ground_truth'])
    time.sleep(6)
    relevance = get_judge_score("Relevance", item['question'], context_text, rag_answer, item['ground_truth'])
    
    results.append({
        "Question": item['question'],
        "RAG Answer": rag_answer,          
        "Ground Truth": item['ground_truth'], 
        "Faithfulness": faithfulness['score'],
        "Relevance": relevance['score'],
        "Notes": faithfulness['reason']
    })

df = pd.DataFrame(results)
print("\n--- FINAL EVALUATION REPORT ---")
print(df[["Question", "Faithfulness", "Relevance"]])
print(f"\nAverage Faithfulness: {df['Faithfulness'].mean():.2f}")
print(f"Average Relevance: {df['Relevance'].mean():.2f}")

df.to_csv("rag_evaluation_results.csv", index=False)