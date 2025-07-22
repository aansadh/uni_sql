"""
This module provides the MCPClient class for interacting with the MCP server.

The MCPClient class is designed to manage communication with the MCP server,
including listing resources, prompts, and tools, as well as calling tools with
specific parameters. It is implemented as an asynchronous context manager to
ensure proper session management.
"""

from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession
from rich import print
from host.core.config import settings
import logging
from typing import List
from mcp import Tool

logger = logging.getLogger(__name__)

class MCPClient:
    """
    A client for interacting with the MCP server.

    Attributes:
        url (str): The MCP server URL.
        _session (ClientSession): The current client session.
        _streameablehttp_client (streamablehttp_client): The HTTP client for streaming.
        _client_session (ClientSession): The client session for communication.
        tools (List[Tool]): A list of tools available on the MCP server.

    NOTE:
        This class is designed to be used as an asynchronous context manager.
        Use `async with MCPClient() as client:` to ensure proper session management.
    """

    def __init__(self, url: str = None):
        """
        Initializes the MCPClient instance.

        Args:
            url (str, optional): The MCP server URL. Defaults to the value in settings.
        """
        self.url = settings.MCP_SERVER_URL.strip() if settings.MCP_SERVER_URL else "http://localhost:8000/mcp"
        self._session = None
        self._streameablehttp_client: streamablehttp_client = None
        self._client_session: ClientSession = None
        self.tools: List[Tool] = []
        logger.info("MCPClient initialized with URL: %s", self.url)

    async def __aenter__(self):
        """
        Asynchronous context manager entry point.

        Initializes the streaming HTTP client and client session.

        Returns:
            MCPClient: The initialized MCPClient instance.
        """
        logger.info("Entering MCPClient context manager.")
        self._streameablehttp_client = streamablehttp_client(url=self.url)
        read_stream, write_stream, _ = await self._streameablehttp_client.__aenter__()
        self._client_session = ClientSession(read_stream, write_stream)
        self._session = await self._client_session.__aenter__()
        await self._session.initialize()
        logger.info("MCP Client session initialized.")
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        """
        Asynchronous context manager exit point.

        Cleans up the client session and streaming HTTP client.

        Args:
            exc_type (type): The exception type, if any.
            exc_value (Exception): The exception instance, if any.
            traceback (Traceback): The traceback object, if any.
        """
        logger.info("Exiting MCPClient context manager.")
        if self._client_session:
            await self._client_session.__aexit__(exc_type, exc_value, traceback)
        if self._streameablehttp_client:
            await self._streameablehttp_client.__aexit__(exc_type, exc_value, traceback)
        logger.info("MCP Client session closed.")

    def _get_session(self):
        """
        Returns the current client session.

        Returns:
            ClientSession: The current client session.

        Raises:
            RuntimeError: If the session is not initialized.
        """
        if not self._session:
            logger.error("Session not initialized. Please use 'async with MCPClient() as client:' to initialize the session.")
            raise RuntimeError("Session not initialized. Please use 'async with MCPClient() as client:' to initialize the session.")
        logger.info("Returning the current client session.")
        return self._session

    async def list_resources(self):
        """
        Asynchronously lists all available resources on the MCP server.

        Returns:
            List[Resource]: A list of available resources.
        """
        logger.info("Listing resources from the MCP server.")
        return await self._get_session().list_resources()

    async def list_prompts(self):
        """
        Asynchronously lists all available prompts on the MCP server.

        Returns:
            List[Prompt]: A list of available prompts.
        """
        logger.info("Listing prompts from the MCP server.")
        return await self._get_session().list_prompts()

    async def list_tools(self):
        """
        Asynchronously lists all available tools on the MCP server.

        Returns:
            List[Tool]: A list of available tools.
        """
        logger.info("Listing tools from the MCP server.")
        tool_list = await self._get_session().list_tools()
        return tool_list.tools

    async def call_tool(self, tool_name: str, arguments: dict):
        """
        Calls a tool on the MCP server with the given parameters.

        Args:
            tool_name (str): The name of the tool to call.
            arguments (dict): The parameters for the tool.

        Returns:
            Any: The result of the tool execution.
        """
        logger.info("Calling tool '%s' with parameters: %s", tool_name, arguments)
        print(f"Calling tool '{tool_name}' with parameters: {arguments}")
        return await self._get_session().call_tool(tool_name, arguments)

async def main():
    """
    Demonstrates the usage of the MCPClient.

    Initializes the client, lists available prompts, tools, and resources,
    and calls a sample tool.
    """
    logger.info("Starting MCPClient main function.")
    async with MCPClient(settings.MCP_SERVER_URL) as client:
        session = await client._get_session()
        print("Session initialized.")

        prompts = await client.list_prompts()
        print(f"Available prompts: {[p.name for p in prompts.prompts]}")

        tools = await client.list_tools()
        print(f"Available tools: {[t.name for t in tools.tools]}")

        resources = await client.list_resources()
        print(f"Available resources: {[r.uri for r in resources.resources]}")

        tool_result = await session.call_tool("execute_sql_query", {"query": "SELECT * FROM customers;"})
        print(f"Tool result: {tool_result}")
    logger.info("MCPClient main function completed.")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())