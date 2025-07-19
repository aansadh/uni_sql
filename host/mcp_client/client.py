from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession
from rich import print
from host.core.config import settings
import logging

logger = logging.getLogger(__name__)

class MCPClient:
    def __init__(self, url: str = None):
        self.url = settings.MCP_SERVER_URL.strip() if settings.MCP_SERVER_URL else "http://localhost:8000/mcp"
        self._session = None
        self._streameablehttp_client: streamablehttp_client = None
        self._client_session: ClientSession = None

    async def __aenter__(self):
        self._streameablehttp_client = streamablehttp_client(url=self.url)
        read_stream, write_stream, _ = await self._streameablehttp_client.__aenter__()
        self._client_session = ClientSession(read_stream, write_stream)
        self._session = await self._client_session.__aenter__()
        await self._session.initialize()
        logger.info("MCP Client session initialized.")
        return self
    
    async def __aexit__(self, exc_type, exc_value, traceback):
        if self._client_session:
            await self._client_session.__aexit__(exc_type, exc_value, traceback)
        if self._streameablehttp_client:
            await self._streameablehttp_client.__aexit__(exc_type, exc_value, traceback)    
    
    async def get_session(self):
        """
        Returns the current client session.

        Returns:
            ClientSession: The current client session.
        """
        if not self._session:
            raise RuntimeError("Session not initialized. Please use 'async with MCPClient() as client:' to initialize the session.")
        return self._session

    async def list_resources(self):
        """
        Lists all available resources on the MCP server.

        Returns:
            List[Resource]: A list of available resources.
        """
        return await self.get_session().list_resources()
    
    async def list_prompts(self):
        """
        Lists all available prompts on the MCP server.

        Returns:
            List[Prompt]: A list of available prompts.
        """
        return await self.get_session().list_prompts()
    
    async def list_tools(self):
        """
        Lists all available tools on the MCP server.

        Returns:
            List[Tool]: A list of available tools.
        """
        return await self.get_session().list_tools()
    
    async def call_tool(self, tool_name: str, params: dict):
        await self._session.call_tool("execute_sql_query", {"query": "SELECT * FROM customers;"})

async def main():
    async with MCPClient(settings.MCP_SERVER_URL) as client:
        session = await client.get_session()
        print("Session initialized.")
        
        prompts = await session.list_prompts()
        print(f"Available prompts: {[p.name for p in prompts.prompts]}")

        tools = await session.list_tools()
        print(f"Available tools: {[t.name for t in tools.tools]}")

        resources = await session.list_resources()
        print(f"Available resources: {[r.uri for r in resources.resources]}")

        tool_result = await session.call_tool("execute_sql_query", {"query": "SELECT * FROM customers;"})
        print(f"Tool result: {tool_result}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())