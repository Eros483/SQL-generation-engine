import os
from urllib.parse import quote_plus
from typing import Literal, List

from langchain_community.utilities import SQLDatabase
from langchain.chat_models import init_chat_model
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from backend.src.prompt_module import (
    select_table_prompt_module, 
    generate_query_prompt_module, 
    query_verification_prompt_module
)
from backend.core.config import settings
from backend.utils.custom_exception import CustomException
from backend.utils.logger import get_logger

logger = get_logger(__name__)

class SQLAgentGenerator:
    """
    ReAct-based Agent for converting natural language to SQL queries using LangGraph.
    
    This class orchestrates a graph workflow that:
    1. Lists available tables.
    2. Selects relevant tables and fetches their schema.
    3. Generates a SQL query.
    4. Verifies the query syntax.
    5. Executes the query and returns the result.
    """

    def __init__(self, api_key: str = None, model_name: str = "google_genai:gemini-2.5-flash-lite"):
        """
        Initialize the agent with API keys, model configuration, and database connection.

        Args:
            api_key (str, optional): Google API Key. Defaults to environment variable.
            model_name (str): The name of the LLM model to use.
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.model_name = model_name
        
        if self.api_key:
            os.environ["GOOGLE_API_KEY"] = self.api_key
        
        self.llm = self._setup_llm()
        self.db = self._setup_database()

        self.tools = self._setup_tools()
        self.tool_map = {tool.name: tool for tool in self.tools}
        
        self.graph = self._build_graph()
        
        logger.info("SQL Agent Initialized")

    def _setup_llm(self):
        """Initialize the ChatModel instance."""
        logger.info(f"Initializing LLM: {self.model_name}")
        return init_chat_model(self.model_name)

    def _setup_database(self) -> SQLDatabase:
        """Establish connection to the SQL database using configuration settings."""
        try:
            encoded_user = quote_plus(settings.DB_USER)
            encoded_password = quote_plus(settings.DB_PASSWORD)
            encoded_name = quote_plus(settings.DB_NAME)

            db_uri = f"mysql+pymysql://{encoded_user}:{encoded_password}@{settings.DB_HOST}/{encoded_name}"
            
            logger.info("Connecting to database...")
            return SQLDatabase.from_uri(
                db_uri,
                sample_rows_in_table_info=0,
                ignore_tables=['DATABASECHANGELOG', 'DATABASECHANGELOGLOCK']
            )
        except Exception as e:
            logger.error("Error in setting up database")
            raise CustomException("Error in setting up database", e)
    
    def _setup_tools(self) -> List:
        """Configure and return the SQL toolkit tools."""
        toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        return toolkit.get_tools()

    def list_tables_node(self, state: MessagesState):
        """Graph Node: Retrieves and lists all available tables in the database."""
        tool_call = {
            "name": "sql_db_list_tables",
            "args": {},
            "id": "init_list_tables",
            "type": "tool_call",
        }
        tool_call_message = AIMessage(content="", tool_calls=[tool_call])
        
        list_tables_tool = self.tool_map["sql_db_list_tables"]
        tool_output = list_tables_tool.invoke({}) 
        
        response_content = f"Available tables: {tool_output}" 
        response = AIMessage(content=response_content)

        return {"messages": [tool_call_message, response]}

    def call_get_schema_node(self, state: MessagesState):
        """Graph Node: Determines which tables are relevant and requests their schema."""
        system_prompt = select_table_prompt_module()
        system_message = {"role": "system", "content": system_prompt}
        
        get_schema_tool = self.tool_map["sql_db_schema"]
        llm_with_tools = self.llm.bind_tools([get_schema_tool], tool_choice="any")
        
        response = llm_with_tools.invoke([system_message] + state["messages"])
        return {"messages": [response]}

    def generate_query_node(self, state: MessagesState):
        """Graph Node: Generates a SQL query based on the schema and user question."""
        system_prompt = generate_query_prompt_module(self.db)
        system_message = {"role": "system", "content": system_prompt}
        
        run_query_tool = self.tool_map["sql_db_query"]
        llm_with_tools = self.llm.bind_tools([run_query_tool])
        
        response = llm_with_tools.invoke([system_message] + state["messages"])
        return {"messages": [response]}

    def check_query_node(self, state: MessagesState):
        """Graph Node: Verifies the generated query syntax and logic before execution."""
        system_prompt = query_verification_prompt_module(self.db)
        system_message = {"role": "system", "content": system_prompt}

        last_message = state["messages"][-1]
        tool_call = last_message.tool_calls[0]
        proposed_query = tool_call["args"].get("query")
        
        user_message = {"role": "user", "content": f"Verify this query: {proposed_query}"}
        
        run_query_tool = self.tool_map["sql_db_query"]
        llm_with_tools = self.llm.bind_tools([run_query_tool], tool_choice="any")
        
        response = llm_with_tools.invoke([system_message, user_message])

        return {"messages": [response]}

    def should_continue(self, state: MessagesState) -> str:
        """Conditional Edge: Determines if the workflow should proceed to query verification or end."""
        messages = state["messages"]
        last_message = messages[-1]
        if not last_message.tool_calls:
            return END
        else:
            return "check_query"

    def _build_graph(self) -> StateGraph:
        """Construct and compile the LangGraph state graph."""
        workflow = StateGraph(MessagesState)

        get_schema_tool = self.tool_map["sql_db_schema"]
        run_query_tool = self.tool_map["sql_db_query"]
        
        workflow.add_node("list_tables", self.list_tables_node)
        workflow.add_node("call_get_schema", self.call_get_schema_node)
        workflow.add_node("get_schema", ToolNode([get_schema_tool]))
        workflow.add_node("generate_query", self.generate_query_node)
        workflow.add_node("check_query", self.check_query_node)
        workflow.add_node("run_query", ToolNode([run_query_tool]))

        workflow.add_edge(START, "list_tables")
        workflow.add_edge("list_tables", "call_get_schema")
        workflow.add_edge("call_get_schema", "get_schema")
        workflow.add_edge("get_schema", "generate_query")
        
        workflow.add_conditional_edges(
            "generate_query",
            self.should_continue,
        )
        
        workflow.add_edge("check_query", "run_query")
        workflow.add_edge("run_query", "generate_query") 

        return workflow.compile()
    
    def run(self, question: str, config: RunnableConfig = None):
            """
            Executes the agent and returns the final response string.
            """
            initial_state = {"messages": [{"role": "user", "content": question}]}
            
            final_step = None
            # Loop through stream to print progress, but capture the last step
            for step in self.graph.stream(initial_state, config=config, stream_mode="values"):
                final_step = step
                
            # Extract the final AI message content
            if final_step:
                return final_step["messages"][-1].content
            return "No response generated."