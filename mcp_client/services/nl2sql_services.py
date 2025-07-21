import logging
from typing import List, Dict
from exceptions import LLMGenerationError # Assuming this path is correct
# from .ollama_services import OllamaServices # Ensure this import is correct as well, if needed

# --- Hardcoded Database Schema (for Week 1 MVP) ---
# IMPORTANT: This matches the schema provided in the Ollama sqlcoder:7b usage example.
DATABASE_SCHEMA_DDL = """
CREATE TABLE products (
  product_id INTEGER PRIMARY KEY, -- Unique ID for each product
  name VARCHAR(50), -- Name of the product
  price DECIMAL(10,2), -- Price of each unit of the product
  quantity INTEGER  -- Current quantity in stock
);

CREATE TABLE customers (
   customer_id INTEGER PRIMARY KEY, -- Unique ID for each customer
   name VARCHAR(50), -- Name of the customer
   address VARCHAR(100) -- Mailing address of the customer
);

CREATE TABLE salespeople (
  salesperson_id INTEGER PRIMARY KEY, -- Unique ID for each salesperson
  name VARCHAR(50), -- Name of the salesperson
  region VARCHAR(50) -- Geographic sales region
);

CREATE TABLE sales (
  sale_id INTEGER PRIMARY KEY, -- Unique ID for each sale
  product_id INTEGER, -- ID of product sold
  customer_id INTEGER,  -- ID of customer who made purchase
  salesperson_id INTEGER, -- ID of salesperson who made the sale
  sale_date DATE, -- Date the sale occurred
  quantity INTEGER -- Quantity of product sold
);

CREATE TABLE product_suppliers (
  supplier_id INTEGER PRIMARY KEY, -- Unique ID for each supplier
  product_id INTEGER, -- Product ID supplied
  supply_price DECIMAL(10,2) -- Unit price charged by supplier
);

-- sales.product_id can be joined with products.product_id
-- sales.customer_id can be joined with customers.customer_id
-- sales.salesperson_id can be joined with salespeople.salesperson_id
-- product_suppliers.product_id can be joined with products.product_id
"""

logger = logging.getLogger(__name__)

class NL2SQLServices:
    """
    A class for interacting with the NL2SQL services to convert natural language queries to SQL.
    """

    def __init__(self, ollama_services): # Type hint OllamaServices for clarity
        """
        Initializes the NL2SQLServices instance.

        Args:
            ollama_services (OllamaServices): The Ollama service instance for generating SQL queries.
        """
        self.ollama_services = ollama_services
        logger.info("NL2SQLServices instance initialized.")

    async def generate_sql_query(self, query: str) -> str:
        """
        Generates a SQL query from a natural language query using the Ollama service.

        Args:
            query (str): The natural language query.

        Returns:
            str: The generated SQL query string.

        Raises:
            LLMGenerationError: If the LLM fails to generate a valid SQL query.
        """
        logger.info(f"Generating SQL for natural language query: '{query}'")

        # --- CRITICAL FIX: Use the EXACT prompt format from [ollama.com/library/sqlcoder:7b](https://ollama.com/library/sqlcoder:7b) ---
        prompt_template = """
        ### Instructions:
        Your task is to convert a question into a SQL query, given a Postgres database schema.
        Adhere to these rules:
        - **Deliberately go through the question and database schema word by word** to appropriately answer the question
        - **Use Table Aliases** to prevent ambiguity. For example, `SELECT table1.col1, table2.col1 FROM table1 JOIN table2 ON table1.id = table2.id`.
        - When creating a ratio, always cast the numerator as float

        ### Input:
        Generate a SQL query that answers the question `{question}`.
        This query will run on a database whose schema is represented in this string:
        {schema_ddl}

        ### Response:
        Based on your instructions, here is the SQL query I have generated to answer the question `{question}`:
        ```sql
        """
        
        # Fill the template with the actual question and schema
        full_prompt = prompt_template.format(
            question=query,
            schema_ddl=DATABASE_SCHEMA_DDL
        )

        # Prepare messages for the Ollama chat model.
        # For sqlcoder, a single user message containing the full prompt is often effective.
        messages: List[Dict[str, str]] = [
            {"role": "user", "content": full_prompt}
        ]

        try:
            llm_response = await self.ollama_services.generate_async(messages=messages)
            
            generated_sql = llm_response.get("content", "").strip()

            sql_start_tag = "```sql"
            sql_end_tag = "```"

            if sql_start_tag in generated_sql:
                # Extract everything after the opening tag
                generated_sql = generated_sql.split(sql_start_tag, 1)[1].strip()
                
                # Check for the closing tag and remove everything after it
                if sql_end_tag in generated_sql:
                    generated_sql = generated_sql.split(sql_end_tag, 1)[0].strip()
                else:
                    # If no closing tag, log a warning, but still try to use what's there
                    logger.warning(f"SQL block started but not properly closed for query: '{query}'")
            else:
                logger.warning(f"No '```sql' block found in LLM response for query: '{query}'. Assuming raw response is SQL.")
                # If the model doesn't use the code block, assume the entire content is the SQL
                pass # generated_sql is already stripped, use it as is if no tag

            # Additional cleaning for comments or unwanted text that might still sneak in
            # These are common patterns for LLMs to add
            generated_sql = generated_sql.split('Comment:')[0].strip()
            generated_sql = generated_sql.split('--')[0].strip()
            generated_sql = generated_sql.split('Explanation:')[0].strip()
            generated_sql = generated_sql.split('Based on your instructions, here is the SQL query')[0].strip() # Remove preamble
            generated_sql = generated_sql.split('```')[0].strip() # Aggressively remove any remaining ```

            generated_sql = generated_sql.strip()

            if not generated_sql:
                logger.warning(f"Ollama returned an empty or invalid SQL query after processing for: '{query}'. Full LLM response was: {llm_response.get('content', '')}")
                raise LLMGenerationError(f"LLM could not generate a valid SQL query for '{query}'.")

            logger.info(f"Successfully generated SQL: '{generated_sql}'")
            return generated_sql

        except LLMGenerationError as e:
            logger.error(f"Failed to generate SQL query for '{query}': {e}", exc_info=True)
            raise # Re-raise the custom exception
        except Exception as e:
            logger.error(f"An unexpected error occurred during SQL generation for '{query}': {e}", exc_info=True)
            raise LLMGenerationError(f"Unexpected error during SQL generation for '{query}': {e}", original_exception=e)