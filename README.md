# Precise_RAG
## Aim:
Assess the performance of different RAG methods in financial documents analysis (primaryly for background investigation for consulting or stock purchase)
## Assessment criteria:
1. relevance
2. length of retrieved context
3. <s>speed
4. <s>cost

## Methods to test:
1. Dense Embeddings
   1.1 parameters
   1.2 <s>finetune embedding model (need GPU machine, too expensive for now)
2. ColBERT
4. Hybrid retriever and rerank
5. <s>Knowledge Augmented Generation (KAG, need to build a domain-specific architecture from sratch)
6. <s>Contextual retrieval preprocessing (use llm to search through all chunks, too expensive)
