import google.generativeai as genai
import numpy as np
import pandas as pd
from .knowledge_base import CONHECIMENTO_BASE

class EmbeddingManager:
    def __init__(self, api_key):
        self.api_key = api_key
        genai.configure(api_key=self.api_key)
        self.model = "models/embedding-001"
        self.documents_df = None
        self._initialize_knowledge_base()
        
    def _initialize_knowledge_base(self):
        """Inicializa a base de conhecimento com o texto predefinido"""
        # Divide o texto em parágrafos ou seções menores
        sections = CONHECIMENTO_BASE.split('\n\n')
        documents = [
            {
                "Titulo": f"Seção {i+1}",
                "Conteudo": section.strip()
            }
            for i, section in enumerate(sections)
            if section.strip()
        ]
        
        self.documents_df = pd.DataFrame(documents)
        self._generate_embeddings()
    
    def _generate_embeddings(self):
        """Gera embeddings para todos os documentos no DataFrame"""
        self.documents_df["Embedding"] = self.documents_df.apply(
            lambda row: self._embed_content(row["Titulo"], row["Conteudo"]),
            axis=1
        )
    
    def _embed_content(self, title, text):
        """Gera embedding para um único documento"""
        response = genai.embed_content(
            model=self.model,
            content=text,
            title=title,
            task_type="RETRIEVAL_DOCUMENT"
        )
        return response["embedding"]
    
    def search_query(self, query):
        """Busca o texto mais relevante para uma consulta"""
        query_embedding = genai.embed_content(
            model=self.model,
            content=query,
            task_type="RETRIEVAL_QUERY"
        )["embedding"]
        
        prod_escalares = np.dot(np.stack(self.documents_df["Embedding"]), query_embedding)
        index = np.argmax(prod_escalares)
        return self.documents_df.iloc[index]["Conteudo"]
