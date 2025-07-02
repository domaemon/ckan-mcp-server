# File agent.py

import asyncio
import json
from typing import Any

from dotenv import load_dotenv
from google.adk.agents.llm_agent import LlmAgent
from google.adk.artifacts.in_memory_artifact_service import (
    InMemoryArtifactService,  # Optional
)
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.mcp_tool.mcp_toolset import (
    MCPToolset,
    SseServerParams,
)
from google.genai import types
from rich import print
load_dotenv()

async def get_tools_async():
    """Gets tools from the File System MCP Server."""
    tools, exit_stack = await MCPToolset.from_server(
        connection_params=SseServerParams(
            url="http://localhost:8001/mcp/sse",
        )
    )
    print("MCP Toolset created successfully.")
    return tools, exit_stack
###
    async def list_tools(self) -> list:
        """
        Requests the list of available tools from the server.
        サーバーから利用可能なツールのリストを要求します。

        Returns:
            list: A list of tool dictionaries.
                  ツールの辞書のリスト。
        """
        logger.info("Requesting 'list_tools' from server...")
        response = await self._send_request("list_tools")
        if "result" in response:
            return response["result"]
        elif "error" in response:
            logger.error(f"Error listing tools: {response['error'].get('message', 'Unknown error')}")
            return []
        return []



###
async def get_agent_async():
    """Creates an ADK Agent equipped with tools from the MCP Server."""
    tools, exit_stack = await get_tools_async()
    print(f"Fetched {len(tools)} tools from MCP server.")
    root_agent = LlmAgent(
        model="gemini-2.0-flash",
        name="assistant",
        instruction="""
        You're a helpful Tokyo Open Dataset assistant. You handle dataset searching. When the user searches for a dataset, mention it's name, id. Always mention dataset ids while performing any searches. This is very important for any operations.
        """,
        tools=tools,
    )
    return root_agent, exit_stack

root_agent = get_agent_async()
