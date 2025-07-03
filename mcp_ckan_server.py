#!/usr/bin/env python3

import asyncio
import json
import logging
import os
import ssl
import certifi
from typing import Annotated, Any, Dict, List, Optional, Union
from pydantic import Field
from urllib.parse import urljoin

import aiohttp
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.routing import Route, Mount

# Import FastMCP and SseServerTransport
from mcp.server.fastmcp import FastMCP
from mcp.shared.exceptions import McpError
from mcp.types import ErrorData, INTERNAL_ERROR, INVALID_PARAMS
from mcp.server.sse import SseServerTransport

from dotenv import load_dotenv

load_dotenv()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# Configure logging
logging.basicConfig(level=logging.INFO, filename="mcp-ckan-server.log", format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("mcp-ckan-server")

class CKANAPIClient:
    """CKAN API client for making HTTP requests"""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = None
    
    async def __aenter__(self):
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        self.session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context))
        logger.info(f"CKANAPIClient session opened for base_url: {self.base_url}")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            logger.info("CKANAPIClient session closed.")
    
    def _get_headers(self) -> Dict[str, str]:
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'MCP-CKAN-Server/1.0'
        }
        if self.api_key:
            headers['Authorization'] = self.api_key
        return headers
    
    async def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        """Make HTTP request to CKAN API"""
        url = urljoin(f"{self.base_url}/api/3/action/", endpoint)
        headers = self._get_headers()
        
        logger.info(f"Making CKAN API request: {method} {url} with data: {data}")
        try:
            async with self.session.request(method, url, headers=headers, json=data) as response:
                result = await response.json()
                # logger.warn(result) # Changed to info, as 'warn' usually indicates a problem, not just data
                logger.info(f"CKAN API response for {endpoint}: {json.dumps(result, indent=2)}")
                if not result.get('success', False):
                    error_msg = result.get('error', {})
                    raise Exception(f"CKAN API Error: {error_msg}")
                
                return result.get('result', {})
        except aiohttp.ClientError as e:
            logger.error(f"HTTP Client Error during request to {url}: {e}")
            raise Exception(f"HTTP Error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during request to {url}: {e}", exc_info=True)
            raise Exception(f"Request failed: {str(e)}")

# Global CKAN client - will be initialized in the lifespan context
ckan_client: Optional[CKANAPIClient] = None

# Initialize FastMCP server
# We use FastMCP directly now, not the lower-level 'Server' class
mcp_server = FastMCP("ckan-mcp-server")

@mcp_server.tool() # Decorator from FastMCP
async def ckan_package_list(
        limit: Annotated[int, Field(description="Maximum number of packages to return")] = 100,
        offset: Annotated[int, Field(description="Offset for pagination")] = 0
) -> List[str]: # Type hints for schema generation
    """Get list of all packages (datasets) in CKAN (unsorted)."""
    if not ckan_client:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message="CKAN client not initialized."))
    
    try:
        result = await ckan_client._make_request("GET", f"package_list?limit={limit}&offset={offset}")
        return result
    except Exception as e:
        logger.error(f"Error in ckan_package_list: {e}")
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to list packages: {e}"))

@mcp_server.tool()
async def ckan_package_show(
        id: Annotated[str, Field(description="Package ID or name")]
) -> Dict[str, Any]:
    """Get details of a specific package/dataset (like dates)."""
    if not ckan_client:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message="CKAN client not initialized."))
    
    try:
        result = await ckan_client._make_request("GET", f"package_show?id={id}")
        return result
    except Exception as e:
        logger.error(f"Error in ckan_package_show: {e}")
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to show package: {e}"))

@mcp_server.tool()
async def ckan_package_search(
        q: Annotated[str, Field(description="Search query")] = "*:*",
        fq: Annotated[str, Field(description="Fliter query")] = None,
        sort: Annotated[str, Field(description="Sort field and direction (e.g., 'score desc')")] = None,
        rows: Annotated[int, Field(description="Number of results to return")] = 10,
        start: Annotated[int, Field(description="Offset for pagination")] = 0
) -> Dict[str, Any]:
    """Search for packages using queries."""
    if not ckan_client:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message="CKAN client not initialized."))
    
    try:
        params = {"q": q, "fq": fq, "sort": sort, "rows": rows, "start": start}
        query_params = []
        for key, value in params.items():
            if value is not None:
                query_params.append(f"{key}={value}")
        query_string = "&".join(query_params)
        result = await ckan_client._make_request("GET", f"package_search?{query_string}")
        return result
    except Exception as e:
        logger.error(f"Error in ckan_package_search: {e}")
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to search packages: {e}"))

@mcp_server.tool()
async def ckan_organization_list(
        all_fields: Annotated[bool, Field(description="Get list of all organizations")] = False
) -> List[Union[str, Dict[str, Any]]]:
    """Get list of all organizations."""
    if not ckan_client:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message="CKAN client not initialized."))
    
    try:
        result = await ckan_client._make_request("GET", f"organization_list?all_fields={all_fields}")
        return result
    except Exception as e:
        logger.error(f"Error in ckan_organization_list: {e}")
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to list organizations: {e}"))

@mcp_server.tool()
async def ckan_organization_show(
        id: Annotated[str, Field(description="Organization ID or name")],
        include_datasets: Annotated[bool, Field(description="Include organization's datasets")] = False,
) -> Dict[str, Any]:
    """Get details of a specific organization."""
    if not ckan_client:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message="CKAN client not initialized."))
    
    try:
        result = await ckan_client._make_request("GET", f"organization_show?id={id}&include_datasets={include_datasets}")
        return result
    except Exception as e:
        logger.error(f"Error in ckan_organization_show: {e}")
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to show organization: {e}"))

@mcp_server.tool()
async def ckan_group_list(
        all_fields: Annotated[bool, Field(description="Include all group fields")] = False
) -> List[Union[str, Dict[str, Any]]]:
    """Get list of all groups."""
    if not ckan_client:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message="CKAN client not initialized."))
    
    try:
        result = await ckan_client._make_request("GET", f"group_list?all_fields={all_fields}")
        return result
    except Exception as e:
        logger.error(f"Error in ckan_group_list: {e}")
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to list groups: {e}"))

@mcp_server.tool()
async def ckan_tag_list(
        vocabulary_id: Annotated[str, Field(description="Vocabulary ID to filter tags")] = None
) -> List[Union[str, Dict[str, Any]]]:
    """Get list of all tags."""
    if not ckan_client:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message="CKAN client not initialized."))
    
    try:
        params = {"vocabulary_id": vocabulary_id}
        query_params = []
        for key, value in params.items():
            if value is not None:
                query_params.append(f"{key}={value}")
        query_string = "&".join(query_params)
        endpoint = f"tag_list?{query_string}" if query_string else "tag_list"
        result = await ckan_client._make_request("GET", endpoint)
        return result
    except Exception as e:
        logger.error(f"Error in ckan_tag_list: {e}")
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to list tags: {e}"))

@mcp_server.tool()
async def ckan_resource_show(
        id: Annotated[str, Field(description="Get details of a specific resource")]
) -> Dict[str, Any]:
    """Get details of a specific resource."""
    if not ckan_client:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message="CKAN client not initialized."))
    
    try:
        result = await ckan_client._make_request("GET", f"resource_show?id={id}")
        return result
    except Exception as e:
        logger.error(f"Error in ckan_resource_show: {e}")
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to show resource: {e}"))

@mcp_server.tool()
async def ckan_site_read() -> Dict[str, Any]:
    """Get site information and statistics."""
    if not ckan_client:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message="CKAN client not initialized."))
    
    try:
        result = await ckan_client._make_request("GET", "site_read")
        return result
    except Exception as e:
        logger.error(f"Error in ckan_site_read: {e}")
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to read site info: {e}"))

@mcp_server.tool()
async def ckan_status_show() -> Dict[str, Any]:
    """Get CKAN site status and version information."""
    if not ckan_client:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message="CKAN client not initialized."))
    
    try:
        result = await ckan_client._make_request("GET", "status_show")
        return result
    except Exception as e:
        logger.error(f"Error in ckan_status_show: {e}")
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to show status: {e}"))

# FastMCP automatically handles listing and calling tools defined with @mcp_server.tool()
# So, handle_list_tools and handle_call_tool are no longer needed as separate functions
# when using FastMCP's decorators.

@mcp_server.resource(uri="ckan://api/docs")
async def ckan_api_docs() -> str:
    """Official CKAN API documentation and endpoints."""
    return """
CKAN API Documentation Summary

Base URL: Configure via CKAN_URL environment variable
API Version: 3

Key Endpoints:
- package_list: Get all packages/datasets
- package_show: Get package details
- package_search: Search packages
- organization_list: Get all organizations  
- organization_show: Get organization details
- group_list: Get all groups
- tag_list: Get all tags
- resource_show: Get resource details
- site_read: Get site information
- status_show: Get site status

Authentication: Set CKAN_API_KEY environment variable for write operations

Full documentation: https://docs.ckan.org/en/latest/api/
    """

@mcp_server.resource(uri="ckan://config")
async def ckan_server_config() -> Dict[str, Any]:
    """Current CKAN server configuration and connection details."""
    config = {
        "base_url": ckan_client.base_url if ckan_client else "Not configured",
        "api_key_configured": bool(ckan_client and ckan_client.api_key),
        "session_active": bool(ckan_client and ckan_client.session)
    }
    return config

# FastMCP automatically handles listing and reading resources defined with @mcp_server.resource()
# So, handle_list_resources and handle_read_resource are no longer needed as separate functions
# when using FastMCP's decorators.

# Setup SSE transport for FastMCP with Starlette
sse_transport = SseServerTransport("/mcp/messages/") # Clients will POST to /mcp/messages/

async def sse_endpoint(request: Request):
    """Handles the Server-Sent Events (GET) connection for MCP communication."""
    # FastMCP uses its internal _mcp_server to run the protocol
    _server_instance = mcp_server._mcp_server
    async with sse_transport.connect_sse(
        request.scope,
        request.receive,
        request._send, # Internal Starlette send function, often used for direct ASGI interaction
    ) as (reader, writer):
        await _server_instance.run(reader, writer, _server_instance.create_initialization_options())

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: Starlette):
    """
    Handles startup and shutdown events for the Starlette application.
    Initializes and cleans up the CKAN API client.
    """
    global ckan_client
    
    ckan_url = os.getenv("CKAN_URL")
    if not ckan_url:
        logger.error("CKAN_URL environment variable not set")
        raise Exception("CKAN_URL environment variable is required to start CKAN MCP Server")
    
    ckan_api_key = os.getenv("CKAN_API_KEY")
    
    ckan_client = CKANAPIClient(ckan_url, ckan_api_key)
    await ckan_client.__aenter__() # Open the aiohttp session
    logger.info("CKAN MCP Server application startup completed.")
    yield # Application runs here
    await ckan_client.__aexit__(None, None, None) # Close the aiohttp session
    logger.info("CKAN MCP Server application shutdown completed.")

app = Starlette(
    debug=True,
    routes=[
        # This route handles the long-lived SSE GET connection from the client
        Route("/mcp/sse", endpoint=sse_endpoint),
        # This route handles the short-lived HTTP POST requests from the client
        # It's managed by sse_transport.handle_post_message
        Mount("/mcp/messages", app=sse_transport.handle_post_message),
    ],
    lifespan=lifespan # Attach the lifespan context manager
)

if __name__ == "__main__":
    host = os.getenv("MCP_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_PORT", "8001"))
    logger.info(f"Starting CKAN MCP Server with FastMCP and Starlette on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
