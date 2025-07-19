from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession
from rich import print

class MCPClient:
    def __init__(self):
        pass

async def main():
    async with streamablehttp_client(url="http://127.0.0.1:8000/mcp") as (
        read_stream,
        write_stream,
        _,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize the connection
            print("Initializing session...")
            await session.initialize()

            print("Session initialized.")
            prompts = await session.list_prompts()
            print(f"Available prompts: {[p.name for p in prompts.prompts]}")

            tools = await session.list_tools()
            print("Available tools:", tools)
            print(f"Available tools: {[t.name for t in tools.tools]}")

            resources = await session.list_resources()
            print(f"Available resources: {[r.uri for r in resources.resources]}")

            tool_result = await session.call_tool("execute_sql_query", {"query": "SELECT * FROM customers;"})

            print(f"Tool result: {tool_result}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())