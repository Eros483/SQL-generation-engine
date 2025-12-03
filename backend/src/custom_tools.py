# ------ Custom SQL Database Tools @ backend/src/custom_tools.py ------
from langchain_core.tools import tool
from langchain_community.utilities import SQLDatabase
from backend.src.rag_manager import SchemaRAG
from backend.src.graph_manager import SchemaGraph
from typing import List, Optional

def get_db_tools(db: SQLDatabase, schema_rag: SchemaRAG, schema_graph: SchemaGraph) -> List:
    """
    Returns a list of custom tools bound to the specific database instance.
    Includes tools for schema inspection, data sampling, relationship discovery, and pathfinding.
    """

    @tool
    def sql_db_query_distinct_values(table_name: str, column_name: str, search_keyword: Optional[str] = None) -> str:
        """
        Finds unique values in a column. 
        CRITICAL: Use 'search_keyword' to filter results if you are looking for specific values.
        
        Args:
            table_name: The table to query.
            column_name: The column to inspect.
            search_keyword: (Optional) A string to filter by (e.g., "Home" will match "Homelessness").
        """
        try:
            if "*" in column_name:
                return "Error: You must specify a specific column name, not *"
                
            if search_keyword:
                # added fuzzy search capability.
                query = f"SELECT DISTINCT {column_name} FROM {table_name} WHERE {column_name} LIKE :keyword LIMIT 15"
                result = db.run(query, parameters={'keyword': f'%{search_keyword}%'})
            else:
                # Default behavior
                result = db.run(f"SELECT DISTINCT {column_name} FROM {table_name} LIMIT 15")
            
            if not result:
                return f"No values found for column '{column_name}' in table '{table_name}' matching '{search_keyword or 'ALL'}'."
                
            formatted_res = f"Distinct values in {table_name}.{column_name}"
            if search_keyword:
                formatted_res += f" (Filtered by '{search_keyword}')"
            
            return f"{formatted_res}:\n{result}"

        except Exception as e:
            return f"Error: {e}"

    @tool
    def sql_db_sample_rows(table_name: str, columns: Optional[str] = "*") -> str:
        """
        Get 3 sample rows from a table.
        
        - BEST for understanding data formatting (e.g., "Is the date '2023-01-01' or '01-Jan-23'?").
        - BEST for checking if a column contains Binary/UUID data (gobbledygook).
        - You can specify columns (e.g., "id, name, created_at") or leave default "*" for all.
        """
        try:
            return db.run(f"SELECT {columns} FROM {table_name} LIMIT 3")
        except Exception as e:
            return f"Error: {e}"

    @tool
    def sql_db_find_relevant_tables(natural_language_query: str) -> str:
        """
        Step 1: Search for relevant tables using Semantic Search (RAG).
        Input should be the concept you are looking for (e.g. "patients in kodiak cohort" or "billing codes").
        Returns table schemas that match the concept.
        """
        return schema_rag.search_tables(natural_language_query, k=4)

    @tool
    def sql_db_find_table_connections(table_names_comma_separated: str) -> str:
        """
        Step 2: Calculates the SQL JOIN path to connect multiple tables.
        
        Use this when:
        1. You have identified 2 or more tables (e.g. 'patient' and 'lob') but don't know the path.
        2. You need to know if a Bridge Table (like 'map_patient_metrics') is required.
        
        Input: A comma-separated list of tables (e.g., "patient, lob").
        Output: The specific SQL 'FROM... JOIN...' clause connecting them.
        """
        tables = [t.strip() for t in table_names_comma_separated.split(',')]
        return schema_graph.find_connection_query(tables)

    @tool
    def sql_db_get_foreign_keys(table_name: str) -> str:
        """
        Finds how a specific table links to others via explicit Foreign Keys.
        Useful for inspecting direct relationships if the Connection Finder fails.
        """
        query = f"""
        SELECT 
            TABLE_NAME, 
            COLUMN_NAME, 
            REFERENCED_TABLE_NAME, 
            REFERENCED_COLUMN_NAME 
        FROM information_schema.KEY_COLUMN_USAGE 
        WHERE TABLE_SCHEMA = DATABASE() 
        AND (TABLE_NAME = '{table_name}' OR REFERENCED_TABLE_NAME = '{table_name}')
        AND REFERENCED_TABLE_NAME IS NOT NULL;
        """
        try:
            result = db.run(query)
            if not result:
                return f"No explicit foreign keys found for {table_name}. You may need to join by matching column names (e.g. patient_id) manually."
            return f"Foreign Key Relationships for {table_name}:\n{result}"
        except Exception as e:
            return f"Error retrieving foreign keys: {e}"

    @tool
    def sql_db_get_column_info(table_name: str) -> str:
        """
        Get technical details about columns (Data Types, Comments).
        Use this to 'Reason' about data types (Integer vs String, Binary vs Text).
        """
        query = f"""
        SELECT COLUMN_NAME, DATA_TYPE, COLUMN_TYPE, COLUMN_COMMENT 
        FROM information_schema.COLUMNS 
        WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = '{table_name}';
        """
        try:
            return db.run(query)
        except Exception as e:
            return f"Error fetching column info: {e}"
    
    @tool
    def sql_db_find_value_location(search_term: str) -> str:
        """
        Global Search: Finds which Table and Column contains a specific string value.
        """
        all_tables = db.get_usable_table_names()
        lookup_tables = [t for t in all_tables if t.endswith(('_type', 'lob', 'screening_type', 'org_attribute_types'))]
        
        found_locations = []
        
        for table in lookup_tables:
            try:
                # FIX: Use _execute with fetch="all" instead of db.run()
                cols_query = f"""
                    SELECT COLUMN_NAME 
                    FROM information_schema.COLUMNS 
                    WHERE TABLE_SCHEMA = DATABASE() 
                    AND TABLE_NAME = '{table}' 
                    AND (DATA_TYPE LIKE '%char%' OR DATA_TYPE LIKE '%text%')
                """
                columns_res = db._execute(cols_query, fetch="all")
                
                for row in columns_res:
                    col = row['COLUMN_NAME']
                    
                    # Check existence
                    check_query = f"SELECT {col} FROM {table} WHERE {col} LIKE %s LIMIT 1"
                    exists = db._execute(check_query, parameters=[f"%{search_term}%"], fetch="one")
                    
                    if exists and exists[col]:
                        found_locations.append(f"Table: {table} | Column: {col} | Example: {exists[col]}")
            except Exception as e:
                continue
                    
        if not found_locations:
            return f"Could not find value '{search_term}' in common lookup tables. Try searching in 'contributor_type' table manually."
            
        return "Value Found in:\n" + "\n".join(found_locations)

    return [
        sql_db_query_distinct_values, 
        sql_db_sample_rows, 
        sql_db_find_relevant_tables, 
        sql_db_find_table_connections,
        sql_db_get_foreign_keys,
        sql_db_get_column_info,
        sql_db_find_value_location
    ]