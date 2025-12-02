# Agentic SQL generator for Caliper
Designed as a Natural Language to SQL engine, to query complex healthcare datasets autonomously. Utilises a multi-agentic workflow, powered by AWS Bedrock (Google Gemini as a fallback client) and LangGraph to reason through database schemas and complex query generation.

Features a Schema Knowledge Graph for finding join paths between distant tables, and RAG system to map user jargon to technical table names.

## Directory Overview
```
backend/
├── core/
│   └── config.py          # Pydantic settings & env var loading
├── schemas/
│   └── chat.py            # API Request/Response models
├── src/
│   ├── agent.py           # Main LangGraph Agent definition
│   ├── custom_tools.py    # Tools exposed to the LLM (pathfinding, etc.)
│   ├── graph_manager.py   # NetworkX Logic for join path discovery
│   ├── prompt_module.py   # System Prompts for different agent states
│   └── rag_manager.py     # FAISS Vector Store for schema search
├── utils/
│   ├── logger.py          # Custom logging configuration
│   └── custom_exception.py
└── main.py                # FastAPI Entry point
```
## Critical Sticking Points that were resolved
| Guardrail               | Description                                                                                                  |
|-------------------------|--------------------------------------------------------------------------------------------------------------|
| Binary UUID Protection  | The database uses BINARY(16). The Agent is trained (and forced via code validation) to wrap these in HEX() to prevent garbled output. |
| Hallucination Check     | The graph_manager physically validates if two tables can be joined before the Agent is allowed to write the SQL. |
| Syntax Correction       | If the Agent writes invalid SQL, the check_query_node catches it and auto-corrects common errors (like missing LIMIT). |
| Data Verification       | The validate_answer_node checks if the result is empty or contains Python byte strings and triggers an automatic retry loop. |
## Tools being used by Agent
| Tool Name                       | Type                     | Key Responsibility                                                                                                     | When the Agent Uses It                                                                                                                                                 |
|---------------------------------|---------------------------|-------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| sql_db_query                    | Execution                | Run SQL. Executes the final SQL query against the database to get actual data.                                         | **CRITICAL:** The final step of any successful workflow. The agent must call this to answer the user.                                                                 |
| sql_db_find_relevant_tables     | Research (RAG)           | Search Schema. Uses vector search (FAISS) to find tables relevant to natural language terms (e.g., "insurance" → lob, map_patient_metrics). | Start of turn. When the user asks a question and the agent doesn't know which tables contain that data.                                                                |
| sql_db_find_table_connections   | Reasoning (Graph)        | Solve Joins. Uses NetworkX/Dijkstra to calculate the shortest valid JOIN path between two or more tables.              | Planning Phase. When the agent knows it needs data from Table A and Table B but doesn't know the foreign keys connecting them.                                        |
| sql_db_schema                   | Research                 | Get Metadata. Returns the CREATE TABLE statement, column names, and data types for specific tables.                    | Validation Phase. To check specific column names (e.g., is it dob or birth_date?) or types (e.g., is id an Integer or Binary?).                                        |
| sql_db_query_distinct_values    | Research                 | Check Content. Selects distinct values from a column to understand formatting.                                         | Filtering Phase. When the user asks for "Active" patients, the agent checks if the status column uses "Active", "ACT", or "1".                                         |
| sql_db_sample_rows              | Research                 | Preview Data. Returns the first 3 rows of a table.                                                                      | Exploration. When the agent wants to see what the data actually "looks like" before writing a complex query.                                                          |
| sql_db_get_foreign_keys         | Reasoning                | Check Links. Lists the foreign keys defined for a specific table.                                                       | Backup Plan. If the Graph tool fails or the agent wants to verify a direct link manually.                                                                              |
| sql_db_get_column_info          | Research                 | Deep Inspection. Detailed info on columns (often redundant with schema, but useful for type checking).                  | Type Safety. Specifically useful for checking if a column is BINARY(16) requiring HEX() conversion.                                                                    |

## Instructions for using the repository
```
git clone https://github.com/Eros483/Caliper-SQL-generator.git
cd Caliper-SQL-generator
cp .env.example .env
```
- Set up env file accordingly.
### Instructions on loading data from sql dump
- Download sql dump data from relevant location.
- The guide assumes the SQL file is named `Dump20250910.sql` and placed in the `data/raw` directory.
```
cd data/raw
sudo apt update
sudo apt install mysql-server mysql-client
sudo systemctl start mysql
sudo mysql_secure_installation
sudo mysql
CREATE USER 'host_name'@'localhost' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON *.* TO 'host_name'@'localhost' WITH GRANT OPTION;
FLUSH PRIVILEGES;
mysql -u host_name -p < Dump20250910.sql
mysql -u host_name -p -N -e \
"SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA='fhs_coredb_local';"

```

### Instructions on loading backend
```
pip install -r requirements.txt
python -m backend.main
```

### Instructions on loading frontend
```
cd frontend
npm install
npm run build
npm run dev
```