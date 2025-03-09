import os
import torch

from langchain.document_loaders import PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings, HuggingFaceEmbeddings
from langchain.retrievers import ParentDocumentRetriever, ContextualCompressionRetriever, EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_chroma import Chroma
from langchain.storage import InMemoryStore
from ragatouille import RAGPretrainedModel

from langchain.llms import HuggingFaceEndpoint, HuggingFaceHub
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate



template = """
<|system|>>
You are a Financial Assistant that follows instructions extremely well.
Please be truthful and give direct answers. Please tell 'I don't know' if user query is not in CONTEXT

CONTEXT: {context}
</s>
<|user|>
{query}
</s>
<|assistant|>
"""

class DenseEmbedRetriever:
    def __init__(self, content, index_name='', parent_chunk_size=800, child_chunk_size=100, embedding_model_name="BAAI/bge-base-en-v1.5"):
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.content = content
        self.index_name = index_name
        
    def build_index(self):
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=self.parent_chunk_size, chunk_overlap=0)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=self.child_chunk_size, chunk_overlap=0)
        vectorstore = Chroma(embedding_function=self.embeddings)
        store = InMemoryStore()
        retriever = ParentDocumentRetriever(
                    vectorstore=vectorstore,
                    docstore=store,
                    child_splitter=child_splitter,
                    parent_splitter=parent_splitter)
        retriever.add_documents(self.content, ids=None)
        return retriever

# colbert
class ColbertRetriever:
    def __init__(self, content, index_name='', max_document_length=512, colbert_k=5, colbert_model_name="colbert-ir/colbertv2.0"):
        self.max_document_length = max_document_length
        self.colbert_k = colbert_k
        self.index_name = index_name
        self.colbert_model_name = colbert_model_name
        self.documents = list(map(lambda x: x.page_content, content))
        self.RAG = RAGPretrainedModel.from_pretrained(colbert_model_name)

    def build_index(self):
        self.RAG.index(
            collection=self.documents,
            index_name=self.index_name,
            max_document_length=self.max_document_length,
            split_documents=True,
        )
        return self.RAG.as_langchain_retriever(k=self.colbert_k)

# bm25
class KeywordRetriever():
    """Currently only bm25 avail"""
    def __init__(self, content, chunk_size=800, chunk_overlap=30, keyword_retriever_k=5):
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                       chunk_overlap=chunk_overlap)
        self.chunks = splitter.split_documents(content)
        self.k = keyword_retriever_k
        
    def build_index(self):
        keyword_retriever = BM25Retriever.from_documents(self.chunks)
        keyword_retriever.k = self.k
        return keyword_retriever

# hybrid
class HybridRetriever():
    def __init__(self, retrievers, retriever_weights):
        self.retrievers = retrievers
        self.retriever_weights = retriever_weights
        
    def build_index(self):
        ensemble_retriever = EnsembleRetriever(retrievers= self.retrievers,
                                               weights=self.retriever_weights)
        return ensemble_retriever

# reranker
class CompressionRetriever():
    def __init__(self, retriever, rerank_top_n=3, rerank_model_name="BAAI/bge-reranker-base"):
        self.rerank_model = HuggingFaceCrossEncoder(model_name=rerank_model_name)
        self.rerank_top_n = rerank_top_n
        self.retriever = retriever
        
    def build(self):
        compressor = CrossEncoderReranker(model=self.rerank_model, 
                                          top_n=self.rerank_top_n)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=self.retriever
        )
        return compression_retriever

class RagChain():
    def __init__(self, retriever, llm, template):
        self.retriever = retriever
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_template(template)

    def build(self):
        output_parser = StrOutputParser()
        chain = (
                 {"context": self.retriever, "query": RunnablePassthrough()}
                    | self.prompt
                    | self.llm
                    | output_parser
                )
        return chain    

class FullChain():
    RETRIEVER_DIC = {
        'Dense': DenseEmbedRetriever,
        'ColBert': ColbertRetriever,
        'Keyword': KeywordRetriever
    }
    def __init__(self, file_path, template=template, retriever_weight_dic={'ColBert':0.7, 'Keyword':0.3}, rerank_top_n=3, chatmodel_name="gpt-4o-mini-2024-07-18"):
        self.llm = ChatOpenAI(model_name=chatmodel_name)
        self.template = template
        self.file_path = file_path
        self.rerank_top_n = rerank_top_n
        self.retriever_weight_dic = retriever_weight_dic

    def load_pdf(self):
        data = PDFMinerLoader(self.file_path, mode='page')
        self.content = data.load()

    def build(self):
        if self.file_path == None:
            nocontext_template = """
                <|system|>>
                    You are a Financial Assistant that follows instructions extremely well.
                    Please be truthful and give direct answers. Please tell 'I don't know' if cannot answer user query.
                </s>
                <|user|>
                {query}
                </s>
                <|assistant|>
                """
            output_parser = StrOutputParser()
            prompt = ChatPromptTemplate.from_template(nocontext_template)
            chain = (
                 {"query": RunnablePassthrough()}
                    | prompt
                    | self.llm
                    | output_parser
                )
            return chain
        self.load_pdf()
        index_name = os.path.basename(self.file_path).split('.')[0]
        retrievers = []
        retriever_weights = []
        for name, weight in self.retriever_weight_dic.items():
            retriever = self.RETRIEVER_DIC[name](self.content).build_index()
            retrievers.append(retriever)
            retriever_weights.append(weight)
        hybrid_r = HybridRetriever(retrievers, retriever_weights).build_index()
        comp_r = CompressionRetriever(hybrid_r, rerank_top_n=self.rerank_top_n).build()
        chain = RagChain(comp_r, self.llm, self.template).build()
        return chain