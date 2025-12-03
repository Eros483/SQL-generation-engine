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
        injected 'Business Context' strings using a compositional tag system.
        This context is critical for helping the LLM understand relationships 
        (e.g., bridge tables) and jargon (e.g., 'LOB' = Insurance).

        Returns:
            list[Document]: A list of LangChain Document objects ready for indexing.
        """
        table_names = self.db.get_usable_table_names()
        documents = []

        logger.info(f"Indexing {len(table_names)} tables for RAG...")

        CONTEXT_LIBRARY = {
            "patient": [
                "MAIN PATIENT TABLE - PRIMARY ENTITY",
                "Contains Demographics: first_name, last_name, middle_name, date_of_birth, sex, race, ethnicity, language",
                "MANDATORY for ANY query asking: 'Who', 'Patient Names', 'List patients', 'Demographics'",
                "Primary Key: patient_id (BINARY(16) UUID - use HEX() when selecting)",
                "Foreign Keys: patient_coordinator (links to user table)",
                "Hub Table: Connects to almost all other tables via patient_id"
            ],
            
            "contributor_type": [
                "MASTER DIAGNOSIS/CONDITION/SDOH DICTIONARY",
                "Contains ALL medical and social conditions in the system",
                "Categories: Mental Health (Anxiety, Depression, PTSD), Chronic Diseases (Diabetes, Hypertension, Asthma), SDOH (Homelessness, Food Insecurity, Housing Instability)",
                "CRITICAL: If user mentions ANY condition/diagnosis/disease BY NAME → YOU MUST USE THIS TABLE",
                "Key Columns:",
                "  - label (VARCHAR): Human-readable name like 'Anxiety', 'Homelessness'",
                "  - contr_type (INT): Primary key/ID for joining",
                "  - is_diag (BOOL): 1 if medical diagnosis, 0 otherwise",
                "  - is_sdoh (BOOL): 1 if social determinant of health",
                "  - org_id (INT): Which organization defined this contributor",
                "Join To: contributor_individual.contr_type = contributor_type.contr_type"
            ],
            
            "contributor_individual": [
                "BRIDGE TABLE: Patient-to-Diagnosis/Condition Link",
                "Purpose: Links individual patients to their specific diagnoses/conditions/SDOH risks",
                "Use Case: 'Which patients have condition X?' or 'Find all patients with Anxiety'",
                "Columns:",
                "  - patient_id (BINARY(16)): Links to patient table",
                "  - contr_type (INT): Links to contributor_type table",
                "  - identified_on_date (DATE): When condition was identified",
                "  - src_label (VARCHAR): Source of identification",
                "Join Path: patient.patient_id = contributor_individual.patient_id AND contributor_individual.contr_type = contributor_type.contr_type"
            ],
            
            "lob": [
                "LINE OF BUSINESS (LOB) = INSURANCE TYPE",
                "Purpose: Represents different insurance plans/types within organizations",
                "Common Values: 'Medicaid', 'Medicare', 'Commercial', 'Managed Care'",
                "Use this table to filter patients by insurance type",
                "Columns:",
                "  - lob_id (BINARY(16)): Primary key",
                "  - name (VARCHAR): Insurance type name",
                "  - org_guid (BINARY(16)): Links to organization",
                "Join Paths:",
                "  - To Patients: lob.lob_id = map_patient_metrics.lob_id",
                "  - To Organization: lob.org_guid = organization.org_guid"
            ],
            
            "map_patient_metrics": [
                "CENTRAL HUB / BRIDGE TABLE",
                "Purpose: Links patients to their insurance (LOB), programs, plans, and provider groups",
                "MANDATORY for any query involving:",
                "  - Insurance/LOB filtering ('Medicaid patients')",
                "  - Organization assignment ('patients in org X')",
                "  - Program enrollment ('patients in program Y')",
                "Columns:",
                "  - patient_id (BINARY(16))",
                "  - lob_id (BINARY(16)): Insurance type",
                "  - group_id (BINARY(16)): Provider group",
                "  - plan_id (BINARY(16)): Specific insurance plan",
                "  - program_id (BINARY(16)): Care management program",
                "  - start_date, end_date: Enrollment period",
                "Join Path: patient -> map_patient_metrics -> lob -> organization"
            ],
            
            "patient_score": [
                "ANALYTICS/RISK SCORES TABLE",
                "Purpose: Contains calculated risk scores and metrics for patients",
                "Use For: 'Highest risk', 'Top scores', 'Risk analysis', 'SDOH scores'",
                "Key Metrics:",
                "  - sdoh_score (DECIMAL): Social determinants of health risk score",
                "  - comp_score (DECIMAL): Comprehensive risk score",
                "  - impactability (DECIMAL): How impactable patient is by interventions",
                "  - risk_contributors_count (TINYINT): Number of risk factors",
                "  - hcc_score (DECIMAL): Hierarchical Condition Category score",
                "  - quality_score (DECIMAL): Quality metrics",
                "  - er_visits, ip_visits (INT): Emergency room and inpatient visits",
                "Columns:",
                "  - patient_id (BINARY(16))",
                "  - calculation_time (DATETIME): When score was calculated",
                "  - service_date_end (DATE): Data period end date",
                "Join: patient.patient_id = patient_score.patient_id",
                "Multiple rows per patient (historical scores) - use MAX(calculation_time) for latest"
            ],
            
            "organization": [
                "MASTER ORGANIZATION TABLE",
                "Purpose: Contains all healthcare organizations in the system",
                "Columns:",
                "  - org_id (INT): Numeric organization identifier",
                "  - org_guid (BINARY(16)): UUID identifier",
                "  - display_name (VARCHAR): Organization name",
                "NEVER join directly to patient table",
                "Must go through: patient -> map_patient_metrics -> lob -> organization",
                "Join Path: lob.org_guid = organization.org_guid"
            ],
            
            "intervention_type": [
                "INTERVENTION DEFINITIONS/CATALOG",
                "Purpose: Dictionary of all intervention types (treatments, services, programs)",
                "Examples: 'Pfizer Vaccine', 'Wellness Visit', 'Care Coordination', 'Mental Health Counseling'",
                "Financial Data:",
                "  - cost_pre (DECIMAL): Cost before intervention",
                "  - cost_delta (DECIMAL): Cost change after intervention",
                "Columns:",
                "  - intrv_type (INT): Primary key",
                "  - label (VARCHAR): Human-readable intervention name",
                "Join To: intervention_service.intrv_type = intervention_type.intrv_type"
            ],
            
            "intervention_service": [
                "INTERVENTION TRANSACTIONS/HISTORY",
                "Purpose: Records of interventions actually delivered to patients",
                "Use For: 'Which patients received intervention X?', 'Intervention history'",
                "Columns:",
                "  - patient_id (BINARY(16))",
                "  - intrv_type (INT): Links to intervention_type",
                "  - service_date (DATE): When intervention was provided",
                "Join Path: patient -> intervention_service -> intervention_type"
            ],
            
            # ==================== QUERY PATTERN TAGS ====================
            "diagnosis_query_pattern": [
                "",
                "╔════════════════════════════════════════════════════════════╗",
                "║  PATTERN: Queries About Medical Conditions/Diagnoses      ║",
                "╚════════════════════════════════════════════════════════════╝",
                "",
                "Triggers: User mentions condition names like 'Anxiety', 'Depression', 'Diabetes', 'Homelessness'",
                "",
                "MANDATORY 3-STEP PROCESS:",
                "",
                "Step 1: VERIFY SPELLING",
                "  - Use sql_db_find_value_location(search_term='Anxiety')",
                "  - Or: sql_db_query_distinct_values(table_name='contributor_type', column_name='label', search_keyword='Anxiety')",
                "  - Confirms exact label in database (might be 'Anxiety Disorder' not 'Anxiety')",
                "",
                "Step 2: BUILD JOIN PATH",
                "  - Required Tables: patient, contributor_individual, contributor_type",
                "  - Join: patient.patient_id = contributor_individual.patient_id",
                "  - Join: contributor_individual.contr_type = contributor_type.contr_type",
                "",
                "Step 3: EXECUTE QUERY",
                "  - Filter: WHERE contributor_type.label LIKE '%Anxiety%'",
                "  - Select: Use HEX(patient.patient_id) or BIN_TO_UUID(patient.patient_id) for patient IDs",
                "",
                "Example Query:",
                "  SELECT p.first_name, BIN_TO_UUID(p.patient_id) as uuid",
                "  FROM patient p",
                "  JOIN contributor_individual ci ON p.patient_id = ci.patient_id",
                "  JOIN contributor_type ct ON ci.contr_type = ct.contr_type",
                "  WHERE ct.label LIKE '%Anxiety%' OR ct.label LIKE '%Depression%'",
                "  LIMIT 10;",
                "",
                "⚠️ DO NOT skip Step 1 - exact spelling matters!"
            ],
            
            "insurance_query_pattern": [
                "",
                "╔════════════════════════════════════════════════════════════╗",
                "║  PATTERN: Queries About Insurance/LOB                      ║",
                "╚════════════════════════════════════════════════════════════╝",
                "",
                "Triggers: 'Medicaid patients', 'Medicare', 'Commercial insurance', 'patients by insurance type'",
                "",
                "MANDATORY 3-STEP PROCESS:",
                "",
                "Step 1: VERIFY INSURANCE NAME",
                "  - Use sql_db_query_distinct_values(table_name='lob', column_name='name', search_keyword='Medicaid')",
                "  - Confirms exact naming (might be 'Medicaid Managed Care' not 'Medicaid')",
                "",
                "Step 2: BUILD JOIN PATH",
                "  - Required Tables: patient, map_patient_metrics, lob",
                "  - Join: patient.patient_id = map_patient_metrics.patient_id",
                "  - Join: map_patient_metrics.lob_id = lob.lob_id",
                "",
                "Step 3: EXECUTE QUERY",
                "  - Filter: WHERE lob.name LIKE '%Medicaid%'",
                "  - Consider: Check date ranges (start_date, end_date) for current enrollment",
                "",
                "Example Query:",
                "  SELECT p.first_name, p.last_name, l.name as insurance_type",
                "  FROM patient p",
                "  JOIN map_patient_metrics mpm ON p.patient_id = mpm.patient_id",
                "  JOIN lob l ON mpm.lob_id = l.lob_id",
                "  WHERE l.name LIKE '%Medicaid%'",
                "  LIMIT 10;"
            ],
            
            "organization_query_pattern": [
                "",
                "╔════════════════════════════════════════════════════════════╗",
                "║  PATTERN: Queries About Organizations                      ║",
                "╚════════════════════════════════════════════════════════════╝",
                "",
                "Triggers: 'Patients in org X', 'org_id 16', 'organization filtering'",
                "",
                "⚠️ CRITICAL PATH (DO NOT SKIP BRIDGE TABLES):",
                "",
                "Full Join Path:",
                "  patient -> map_patient_metrics -> lob -> organization",
                "",
                "Required Joins:",
                "  - patient.patient_id = map_patient_metrics.patient_id",
                "  - map_patient_metrics.lob_id = lob.lob_id",
                "  - lob.org_guid = organization.org_guid",
                "",
                "Filter Options:",
                "  - WHERE organization.org_id = 16",
                "  - WHERE organization.display_name LIKE '%CareSouth%'",
                "",
                "Example Query:",
                "  SELECT p.first_name, o.display_name",
                "  FROM patient p",
                "  JOIN map_patient_metrics mpm ON p.patient_id = mpm.patient_id",
                "  JOIN lob l ON mpm.lob_id = l.lob_id",
                "  JOIN organization o ON l.org_guid = o.org_guid",
                "  WHERE o.org_id = 16",
                "  LIMIT 10;"
            ],
            
            "risk_score_query_pattern": [
                "",
                "╔════════════════════════════════════════════════════════════╗",
                "║  PATTERN: Queries About Risk Scores/Analytics             ║",
                "╚════════════════════════════════════════════════════════════╝",
                "",
                "Triggers: 'Highest SDOH score', 'Top risk patients', 'Most impactable'",
                "",
                "Key Points:",
                "  - Use patient_score table for all score-based queries",
                "  - Multiple scores per patient (historical) - usually want latest",
                "  - Common pattern: ORDER BY score DESC + LIMIT N",
                "",
                "Getting Latest Scores:",
                "  - Subquery: SELECT MAX(calculation_time) ... GROUP BY patient_id",
                "  - Or window function: ROW_NUMBER() OVER (PARTITION BY patient_id ORDER BY calculation_time DESC)",
                "",
                "Example Query (Top 5 SDOH):",
                "  SELECT p.first_name, ps.sdoh_score",
                "  FROM patient p",
                "  JOIN patient_score ps ON p.patient_id = ps.patient_id",
                "  WHERE ps.calculation_time = (",
                "    SELECT MAX(calculation_time)",
                "    FROM patient_score ps2",
                "    WHERE ps2.patient_id = ps.patient_id",
                "  )",
                "  ORDER BY ps.sdoh_score DESC",
                "  LIMIT 5;"
            ],
            
            "combined_diagnosis_org_pattern": [
                "",
                "╔════════════════════════════════════════════════════════════╗",
                "║  PATTERN: Diagnosis + Organization Filter (COMPLEX)       ║",
                "╚════════════════════════════════════════════════════════════╝",
                "",
                "Triggers: 'Patients with Anxiety in org 16', 'Homelessness patients from organization X'",
                "",
                "⚠️ This requires BOTH diagnosis and organization join paths",
                "",
                "STEP-BY-STEP:",
                "",
                "Step 1: Verify condition spelling in contributor_type",
                "Step 2: Verify organization exists and has that condition defined",
                "  - Important: contributor_type.org_id must match organization.org_id",
                "  - Use: SELECT org_id FROM contributor_type WHERE label LIKE '%Homelessness%'",
                "",
                "Step 3: Build complete join path:",
                "  patient",
                "  ├─> contributor_individual (diagnosis link)",
                "  │   └─> contributor_type (condition definition)",
                "  └─> map_patient_metrics (org link)",
                "      └─> lob",
                "          └─> organization",
                "",
                "Example Query:",
                "  SELECT DISTINCT p.first_name, p.last_name, ct.label as condition",
                "  FROM patient p",
                "  JOIN contributor_individual ci ON p.patient_id = ci.patient_id",
                "  JOIN contributor_type ct ON ci.contr_type = ct.contr_type",
                "  JOIN map_patient_metrics mpm ON p.patient_id = mpm.patient_id",
                "  JOIN lob l ON mpm.lob_id = l.lob_id",
                "  JOIN organization o ON l.org_guid = o.org_guid",
                "  WHERE ct.label LIKE '%Homelessness%'",
                "    AND o.org_id = 16",
                "  LIMIT 10;",
                "",
                "⚠️ Note: contributor_type.org_id and organization.org_id are separate concepts!",
                "  - contributor_type.org_id = who DEFINED the condition",
                "  - Filter by organization.org_id = which org the PATIENT belongs to"
            ]
        }

        TABLE_TO_CONTEXTS = {
            "patient": [
                "patient", 
                "diagnosis_query_pattern", 
                "insurance_query_pattern",
                "organization_query_pattern",
                "risk_score_query_pattern"
            ],
            "contributor_type": [
                "contributor_type", 
                "diagnosis_query_pattern",
                "combined_diagnosis_org_pattern"
            ],
            "contributor_individual": [
                "contributor_individual", 
                "diagnosis_query_pattern",
                "combined_diagnosis_org_pattern"
            ],
            "lob": [
                "lob", 
                "insurance_query_pattern",
                "organization_query_pattern"
            ],
            "map_patient_metrics": [
                "map_patient_metrics", 
                "insurance_query_pattern",
                "organization_query_pattern",
                "combined_diagnosis_org_pattern"
            ],
            "patient_score": [
                "patient_score",
                "risk_score_query_pattern"
            ],
            "organization": [
                "organization",
                "organization_query_pattern",
                "combined_diagnosis_org_pattern"
            ],
            "intervention_type": [
                "intervention_type"
            ],
            "intervention_service": [
                "intervention_service"
            ]
        }
        
        for table in table_names:
            schema = self.db.get_table_info([table])

            context_parts = []
            if table in TABLE_TO_CONTEXTS:
                for tag in TABLE_TO_CONTEXTS[table]:
                    if tag in CONTEXT_LIBRARY:
                        context_parts.extend(CONTEXT_LIBRARY[tag])

            if not context_parts:
                context_parts = [f"Standard table: {table} (no special context defined)"]

            final_context = "\n".join(context_parts)
            
            content = f"""
    ═══════════════════════════════════════════════════════════════
    TABLE: {table}
    ═══════════════════════════════════════════════════════════════

    BUSINESS CONTEXT:
    {final_context}

    ───────────────────────────────────────────────────────────────
    TECHNICAL SCHEMA:
    ───────────────────────────────────────────────────────────────
    {schema}
    """
            
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