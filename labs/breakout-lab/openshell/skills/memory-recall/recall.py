#!/usr/bin/env python3
"""
memory-recall skill — MCP client wrapper (runs INSIDE the OpenShell sandbox).

This is the in-sandbox surface exposed to the OpenClaw agent. It forwards
tool calls over the MCP protocol to the Memory MCP server running OUTSIDE the
sandbox. Because the server forwards `user_id`/`query` into vulnerable sinks,
this skill is also the breakout entry point (see repo `exploits/`).

Usage:
    python recall.py --tool memory_agent --query "..." --user-id ruth
    python recall.py --tool fetch_memory_items --query "=..." --user-id ruth

Environment:
    MEMORY_MCP_URL   URL of the Memory MCP server
                     (default: http://host.openshell.internal:8999/mcp)
"""

import argparse
import asyncio
import os
import sys

from fastmcp import Client

DEFAULT_URL = os.environ.get(
    "MEMORY_MCP_URL", "http://host.openshell.internal:8999/mcp"
)

VALID_TOOLS = {"memory_agent", "restart_memory_agent", "fetch_memory_items"}


async def call(url: str, tool: str, query: str, user_id: str):
    async with Client(url) as client:
        result = await client.call_tool(
            tool, {"query": query, "user_id": user_id}
        )
        return result


def main():
    parser = argparse.ArgumentParser(description="memory-recall MCP client skill")
    parser.add_argument("--tool", default="memory_agent", choices=sorted(VALID_TOOLS))
    parser.add_argument("--query", required=True)
    parser.add_argument("--user-id", default="ruth")
    parser.add_argument("--url", default=DEFAULT_URL)
    args = parser.parse_args()

    print(f"[memory-recall] server={args.url} tool={args.tool}", file=sys.stderr)
    try:
        result = asyncio.run(call(args.url, args.tool, args.query, args.user_id))
    except Exception as exc:
        print(f"[memory-recall] error: {exc}", file=sys.stderr)
        sys.exit(1)

    # Print whatever the server returned so the agent can read it.
    try:
        for block in result.content:
            text = getattr(block, "text", None)
            if text is not None:
                print(text)
    except Exception:
        print(result)


if __name__ == "__main__":
    main()
