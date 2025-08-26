import torch
import warnings
from typing import List
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from ..types import RerankResult
from ..config.settings import settings

warnings.filterwarnings("ignore", message="flash_attn is not installed")


class LocalJinaReranker:
    def __init__(self, model_name: str = None, top_n: int = None):
        self.model_name = model_name or settings.RERANKER_MODEL
        self.top_n = top_n or settings.RERANKER_TOP_N
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = CrossEncoder(
            self.model_name,
            model_kwargs={"torch_dtype": "auto"},
            trust_remote_code=True,
            device=self.device
        )
    
    def rerank(self, query: str, documents: List[Document]) -> RerankResult:
        if not documents:
            return RerankResult(
                documents=[],
                original_count=0,
                final_count=0
            )
        
        original_count = len(documents)
        
        doc_texts = [doc.page_content for doc in documents]
        
        rankings = self.model.rank(
            query, 
            doc_texts, 
            return_documents=True, 
            top_k=min(self.top_n, len(documents))
        )
        
        reranked_docs = []
        for ranking in rankings:
            doc_index = ranking['corpus_id']
            reranked_docs.append(documents[doc_index])
        
        return RerankResult(
            documents=reranked_docs,
            original_count=original_count,
            final_count=len(reranked_docs)
        )
    
    def compress_documents(self, documents: List[Document], query: str) -> List[Document]:
        result = self.rerank(query, documents)
        return result.documents


def create_reranker(model_name: str = None, top_n: int = None) -> LocalJinaReranker:
    return LocalJinaReranker(model_name=model_name, top_n=top_n)


def rerank_documents(query: str, documents: List[Document], reranker: LocalJinaReranker = None) -> RerankResult:
    if reranker is None:
        reranker = create_reranker()
    
    return reranker.rerank(query, documents)
