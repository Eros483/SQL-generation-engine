# ----- Schema RAG Manager @ backend/src/rag_manager.py ------

import os
from langchain_community.utilities import SQLDatabase
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from backend.utils.logger import get_logger
from backend.core.config import settings

logger = get_logger(__name__)


class SchemaRAG:
    """
    Manages semantic search over the database schema.
    
    This class indexes table schemas and manual 'business dictionary' descriptions 
    into a vector store (FAISS). It allows the agent to find relevant tables 
    based on natural language queries, bridging the gap between user terminology 
    and technical table names.
    """

    def __init__(self, db: SQLDatabase):
        """
        Initialize the SchemaRAG manager.

        Args:
            db (SQLDatabase): The database instance to inspect.
        """
        self.db = db
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=settings.GEMINI_API_KEY
        )
        self.vector_store = None
        self._build_index()

    def _get_table_info(self):
        """
        Scans the database and constructs semantic documents for each table.
        
        This method combines the raw SQL schema (columns/types) with manually 
        injected 'Business Context' strings. This context is critical for 
        helping the LLM understand relationships (e.g., bridge tables) and 
        jargon (e.g., 'LOB' = Insurance).

        Returns:
            list[Document]: A list of LangChain Document objects ready for indexing.
        """
        table_names = self.db.get_usable_table_names()
        documents = []

        logger.info(f"Indexing {len(table_names)} tables for RAG...")

        for table in table_names:
            schema = self.db.get_table_info([table])
            
            extra_context = ""
            
            if table == "patient":
                extra_context = "MAIN PATIENT TABLE. Contains Demographics: first_name, last_name, dob. MANDATORY JOIN for any query asking for 'Patient Names' or 'Who'. JOIN: Connects to almost all other tables via `patient_id`."
            elif table == "contributor_individual":
                extra_context = "BRIDGE TABLE. Links Patients to SDOH/Risks/Contributors. Join: patient.patient_id = contributor_individual.patient_id."
            elif table == "map_patient_metrics":
                extra_context = "CENTRAL HUB / BRIDGE. Mandatory for linking Patients to Insurance (LOB). JOIN PATH: patient (patient_id) <-> map_patient_metrics (lob_id) <-> lob."
            elif table == "lob":
                extra_context = "Line of Business (LOB). Represents Insurance Types: Medicaid, Medicare, Commercial. Filter here for 'Medicaid' patients. Join to map_patient_metrics via lob_id."
            elif table == "patient_score":
                extra_context = "ANALYTICS. Contains: sdoh_score, risk_contributors_count. Join: patient.patient_id = patient_score.patient_id."
            elif table == "intervention_type":
                extra_context = "Definitions of Interventions. Financials: cost_pre, cost_delta. Contains Labels like 'Pfizer', 'Wellness Visit'."
            elif table == "intervention_service":
                extra_context = "TRANSACTIONS. Records Interventions given to patients. Link to intervention_type via intrv_type."
            
            content = f"Table: {table}\nDescription: {extra_context}\nSchema:\n{schema}"
            
            documents.append(Document(
                page_content=content,
                metadata={"table_name": table}
            ))
        
        return documents

    def _build_index(self):
        """
        Builds the FAISS vector index from the table documents.
        """
        try:
            documents = self._get_table_info()
            if documents:
                self.vector_store = FAISS.from_documents(documents, self.embeddings)
                logger.info("Schema RAG Index created successfully.")
            else:
                logger.warning("No tables found to index.")
        except Exception as e:
            logger.error(f"Failed to build RAG index: {e}")

    def search_tables(self, query: str, k: int = 5) -> str:
        """
        Performs a semantic similarity search to find relevant tables.

        Args:
            query (str): The user's natural language question.
            k (int): Number of table results to return (default 5 to capture bridge tables).

        Returns:
            str: A formatted string containing the schema and description of the top matching tables.
        """
        if not self.vector_store:
            return "Error: Vector store not initialized."

        results = self.vector_store.similarity_search(query, k=k)
        
        output = []
        for doc in results:
            output.append(doc.page_content)
            
        return "\n\n".join(output)