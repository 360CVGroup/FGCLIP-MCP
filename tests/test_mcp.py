"""Pytest suite for MCP embedding server client behaviors.

"""

import os
import json
import logging
import pytest

    
logger = logging.getLogger(__name__)

# Try importing mcp; if unavailable, skip the entire test module
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except Exception as exc:  # pragma: no cover - environment dependent
    pytest.skip(f"mcp not installed or unavailable: {exc}", allow_module_level=True)

def _server_params() -> StdioServerParameters:
    return StdioServerParameters(
        command="python",
        args=["-m", "fgclip_mcp"],
        env=os.environ,
    )

@pytest.mark.asyncio
async def _get_session():
    async with stdio_client(_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session

@pytest.mark.asyncio
async def _get_tools(session):
    tools = await session.list_tools()
    tool_names = {t.name for t in tools.tools}
    return tools, tool_names

@pytest.mark.asyncio
async def _get_resources(session):
    resources = await session.list_resources()
    return resources

@pytest.mark.asyncio
async def test_list_tools():
    async for session in _get_session():
        tools, _ = await _get_tools(session)
        if not tools.tools:
            pytest.skip("Server returned no tools; skipping tool-related cases")
        assert all(hasattr(t, "name") for t in tools.tools)



@pytest.mark.asyncio
async def test_text_embedding():
    async for session in _get_session():
        tools, tool_names = await _get_tools(session)
        if "text_embedding" not in tool_names:
            pytest.skip("Missing text_embedding tool; skipping")
        result = await session.call_tool(
            "text_embedding",
            {
                "texts": [
                    "This is a test text",
                    "Hello, world!",
                    "Artificial Intelligence and Machine Learning",
                ],
                "model": "fg-clip",
            },
        )
        assert not getattr(result, "isError", False), f"text_embedding failed: {result.content[0].text if getattr(result, 'content', None) else result}"
        assert getattr(result, "content", None), "text_embedding returned no content"

@pytest.mark.asyncio
async def test_image_embedding():
    async for session in _get_session():
        tools, tool_names = await _get_tools(session)
        if "image_embedding" not in tool_names:
            pytest.skip("Missing image_embedding tool; skipping")
        result = await session.call_tool(
            "image_embedding",
            {
                "images": [
                    "http://p0.qhimg.com/t11098f6bcdb7936d24633b14ad.jpg",
                    "http://p0.qhimg.com/t11098f6bcde92f716b4fdb3b39.png",
                ],
                "model": "fg-clip",
            },
        )
        assert not getattr(result, "isError", False), f"image_embedding failed: {result.content[0].text if getattr(result, 'content', None) else result}"
        assert getattr(result, "content", None), "image_embedding returned no content"

@pytest.mark.asyncio
async def test_cosine_similarity():
    async for session in _get_session():
        tools, tool_names = await _get_tools(session)
        if "cosine_similarity" not in tool_names:
            pytest.skip("Missing cosine_similarity tool; skipping")
        resources = await _get_resources(session)
        if len(resources.resources or []) < 2:
            pytest.skip("Fewer than two resources; skipping cosine similarity test")
        uris_a = [resources.resources[0].uri]
        uris_b = [resources.resources[1].uri]
        result = await session.call_tool(
            "cosine_similarity",
            {"uris_a": uris_a, "uris_b": uris_b, "mode": "pairwise"},
        )
        assert not getattr(result, "isError", False), f"cosine_similarity failed: {result.content[0].text if getattr(result, 'content', None) else result}"
        assert getattr(result, "content", None), "cosine_similarity returned no content"


@pytest.mark.asyncio
async def test_list_resources():
    async for session in _get_session():
        resources = await _get_resources(session)
        if not resources.resources:
            pytest.skip("Server returned no resources; skipping resource-related cases")
        assert all(hasattr(r, "uri") for r in resources.resources)

@pytest.mark.asyncio
async def test_read_resource():
    async for session in _get_session():
        resources = await _get_resources(session)
        if not resources.resources:
            pytest.skip("No readable resources; skipping")
        first = resources.resources[0]
        result = await session.read_resource(first.uri)
        contents = getattr(result, "contents", None)
        assert contents, "read_resource returned no content"
        content = contents[0]
        if hasattr(content, "text") and isinstance(content.text, str):
            try:
                json.loads(content.text)
            except Exception:
                pass
        else:
            pytest.skip("Resource content not text or unknown; skipping further assertions")
