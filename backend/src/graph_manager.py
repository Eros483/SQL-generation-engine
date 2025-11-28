import networkx as nx
from langchain_community.utilities import SQLDatabase
from backend.utils.logger import get_logger

logger = get_logger(__name__)

class SchemaGraph:
    def __init__(self, db: SQLDatabase):
        self.db = db
        self.graph = nx.Graph() 
        self._build_graph()

    def _build_graph(self):
        """
        1. Auto-discovers Foreign Keys from the DB.
        2. Allows manual injection of 'Logical' joins that are missing in the DB schema.
        """
        try:
            # --- 1. AUTOMATIC DISCOVERY (MySQL/MariaDB) ---
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
                # Add edge: t1 <-> t2
                self.graph.add_edge(t1, t2, on=f"{t1}.{c1} = {t2}.{c2}")

            # --- 2. MANUAL BRIDGES (The "Secret Sauce") ---
            # If your DB is missing keys (e.g. Binary IDs), add them here manually.
            # Format: (Table A, Table B, Join Condition)
            manual_edges = [
                ("patient", "map_patient_metrics", "patient.patient_id = map_patient_metrics.patient_id"),
                ("map_patient_metrics", "lob", "map_patient_metrics.lob_id = lob.lob_id"),
                ("patient", "patient_score", "patient.patient_id = patient_score.patient_id"),
                ("patient", "contributor_individual", "patient.patient_id = contributor_individual.patient_id"),
                ("contributor_individual", "contributor_type", "contributor_individual.contr_type = contributor_type.contr_type"),
                ("intervention_service", "intervention_type", "intervention_service.intrv_type = intervention_type.intrv_type")
            ]

            for t1, t2, condition in manual_edges:
                self.graph.add_edge(t1, t2, on=condition)

            logger.info(f"Schema Graph built: {self.graph.number_of_nodes()} tables, {self.graph.number_of_edges()} links.")

        except Exception as e:
            logger.error(f"Graph build failed: {e}")

    def find_connection_query(self, table_names: list) -> str:
        """
        Input: ['patient', 'lob']
        Output: "To connect these, join: patient -> map_patient_metrics -> lob"
        """
        if not table_names or len(table_names) < 2:
            return "Need at least 2 tables to find a connection."

        # Filter out tables that don't exist in our graph
        valid_tables = [t for t in table_names if t in self.graph.nodes]
        if len(valid_tables) < 2:
            return f"Cannot connect. Tables not found in graph: {set(table_names) - set(self.graph.nodes)}"

        try:
            # We treat the first table as the 'Anchor' (e.g. Patient)
            # and try to connect everything else to it.
            start = valid_tables[0]
            targets = valid_tables[1:]
            
            full_path_logic = []
            
            for target in targets:
                # Use Dijkstra to find shortest hop
                path = nx.shortest_path(self.graph, source=start, target=target)
                
                # Convert list [A, B, C] into "JOIN B on... JOIN C on..."
                for i in range(len(path) - 1):
                    t_a = path[i]
                    t_b = path[i+1]
                    condition = self.graph[t_a][t_b]['on']
                    full_path_logic.append(f"JOIN {t_b} ON {condition}")

            # Deduplicate joins (in case paths overlap)
            unique_joins = sorted(list(set(full_path_logic)))
            
            return f"FROM {start}\n" + "\n".join(unique_joins)

        except nx.NetworkXNoPath:
            return "No path found. These tables appear to be disconnected."
        except Exception as e:
            return f"Pathfinding error: {e}"