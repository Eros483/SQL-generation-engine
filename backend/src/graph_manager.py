# ----- Schema Graph Manager @ backend/src/graph_manager.py ------

import networkx as nx
from langchain_community.utilities import SQLDatabase
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class SchemaGraph:
    """
    Manages a graph representation of the database schema to facilitate pathfinding
    between tables. It combines automatically discovered foreign keys with manually
    defined logical relationships to map out table connectivity.
    """

    def __init__(self, db: SQLDatabase):
        """
        Initialize the SchemaGraph with a database connection and build the graph.

        Args:
            db (SQLDatabase): The LangChain SQLDatabase instance used to inspect the schema.
        """
        self.db = db
        self.graph = nx.Graph()
        self._build_graph()

    def _build_graph(self):
        """
        Constructs the internal NetworkX graph.
        
        This method performs two main actions:
        1. Queries the database information schema to automatically discover existing foreign keys.
        2. Injects manually defined edges for logical relationships that are not explicitly enforced in the database.
        """
        try:
            fk_query = """
            SELECT 
                TABLE_NAME, COLUMN_NAME, 
                REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME
            FROM information_schema.KEY_COLUMN_USAGE
            WHERE TABLE_SCHEMA = DATABASE() AND REFERENCED_TABLE_NAME IS NOT NULL;
            """
            results = self.db._execute(fk_query, fetch="all")

            for row in results:
                t1, c1 = row['TABLE_NAME'], row['COLUMN_NAME']
                t2, c2 = row['REFERENCED_TABLE_NAME'], row['REFERENCED_COLUMN_NAME']
                
                # DEFAULT WEIGHT = 1.0 (Strong)
                weight = 1.0
                
                # PENALIZING WEAK JOINS (User/Audit logs)
                if "user" in t1 or "user" in t2:
                    weight = 10.0 
                
                self.graph.add_edge(t1, t2, on=f"{t1}.{c1} = {t2}.{c2}", weight=weight)
            manual_edges = [
                ("patient", "map_patient_metrics", "patient.patient_id = map_patient_metrics.patient_id", 1.0),
                ("map_patient_metrics", "lob", "map_patient_metrics.lob_id = lob.lob_id", 1.0),
                ("lob", "organization", "lob.org_guid = organization.org_guid", 1.0), 
                ("patient", "patient_score", "patient.patient_id = patient_score.patient_id", 1.0),
                ("patient", "contributor_individual", "patient.patient_id = contributor_individual.patient_id", 1.0),
                ("contributor_individual", "contributor_type", "contributor_individual.contr_type = contributor_type.contr_type", 1.0),

                ("patient", "user", "patient.patient_coordinator = user.user_id", 10.0)
            ]

            for t1, t2, condition, w in manual_edges:
                self.graph.add_edge(t1, t2, on=condition, weight=w)

        except Exception as e:
            logger.error(f"Graph build failed: {e}")

    def find_connection_query(self, table_names: list) -> str:
        """
        Determines the optimal SQL JOIN clauses required to connect a list of tables.

        Uses Dijkstra's shortest path algorithm to find the sequence of joins 
        connecting the first table in the list to all subsequent tables.

        Args:
            table_names (list): A list of strings representing the table names to connect.

        Returns:
            str: A string containing the 'FROM' clause and necessary 'JOIN' clauses, 
                 or an error message if connectivity cannot be established.
        """
        if not table_names or len(table_names) < 2:
            return "Need at least 2 tables to find a connection."

        valid_tables = [t for t in table_names if t in self.graph.nodes]
        if len(valid_tables) < 2:
            return f"Cannot connect. Tables not found in graph: {set(table_names) - set(self.graph.nodes)}"

        try:
            start = valid_tables[0]
            # HEURISTIC: Always try to anchor on 'patient' if present, to ensure star-schema validity
            if "patient" in valid_tables:
                start = "patient"
                
            full_joins = set()
            
            for target in valid_tables:
                if target == start: continue
                
                # USE WEIGHTS FOR SHORTEST PATH
                path = nx.shortest_path(self.graph, source=start, target=target, weight='weight')

                for i in range(len(path) - 1):
                    t_a = path[i]
                    t_b = path[i+1]
                    edge = self.graph[t_a][t_b]
                    full_joins.add(f"JOIN {t_b} ON {edge['on']}")

            return f"FROM {start}\n" + "\n".join(sorted(list(full_joins)))

        except nx.NetworkXNoPath:
            return "No path found. These tables appear to be disconnected."
        except Exception as e:
            return f"Pathfinding error: {e}"