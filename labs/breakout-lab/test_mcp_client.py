import asyncio
import sys
from fastmcp import Client
from fastmcp.tools import Tool
from colorama import Fore, init

init(autoreset=True)

MCP_SERVER_URL = "http://0.0.0.0:8999/mcp"

def stream_print(text: str):
    """Print text word-by-word to simulate streaming output."""
    words = text.split(" ")
    for i, word in enumerate(words):
        print(Fore.GREEN + word, end="" if i == len(words) - 1 else " ", flush=True)
    print()  # final newline

async def main(query, user_id):
    # Pass URL directly — FastMCP infers StreamableHttpTransport automatically
    async with Client(MCP_SERVER_URL) as client:
        result = await client.call_tool(
            "memory_agent",
            {
                "query": query,
                "user_id": user_id
            }
        )
        output = result.content[0].text
    return output

query = input("Enter your query:\n")
user_id = "ruth"
output = asyncio.run(main(query, user_id))
print("\n")
print(Fore.CYAN + "Assistant: ", end="")
stream_print(output)
