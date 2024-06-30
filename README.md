## Advanced Retrieval Augmented Generation (RAG) chatbot to ask questions about the rules of Cricket!
  - LangChain, Streamlit, and OpenAI API
  - Query rewriting (pre-retrieval) and prompt compression (post-retrieval)
  - Chroma database as external vectorstore
  - Ragas metrics to evaluate system
     - Answer Relevancy score of ~97%, Context Precision score of ~97%, and Context Recall score of 100%
   
  Note: Can be easily modified to Q&A chatbot for any set of documents.

Preview:
  - Only answers questions relating to documents, and does not hallucinate answers to unrelated questions: 
![Advanced_RAG_demo](https://github.com/asvch/ask_cricket/assets/66492476/5118cfc5-60a2-415b-98bf-6053d7ac702d)


All PDF documents in data folder containing the rules of Cricket are from https://www.mumbaicricket.com/mca/Laws_of_Cricket.php, and more documents can be added to folder to expand knowledge of application.
