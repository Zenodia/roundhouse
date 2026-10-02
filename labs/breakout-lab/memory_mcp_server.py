from fastmcp import FastMCP
from dotenv import load_dotenv
from langchain_core.runnables import  RunnablePassthrough
import os
import re
import nest_asyncio, asyncio
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, NVIDIARerank
from langchain_openai import ChatOpenAI
from utils import MemoryOps
import os
from colorama import Fore
load_dotenv('/workspace/.env', override=True)

def strip_think_tags(text: str) -> str:
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

llm = ChatOpenAI(
    model="meta/llama-3.2-11b-vision-instruct",
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ.get("NVIDIA_API_KEY"),
)

embed = NVIDIAEmbeddings(model="nvidia/nemotron-3-embed-1b", truncate="NONE")
if os.getenv("stream") :
    stream_flag = os.getenv("stream")
    if stream_flag.lower()=="yes":
        use_streaming = True
    else:
        use_streaming = False
    print("using streaming : ", os.getenv("stream"))
else:
    use_streaming = False
    print("no environment variable set")

memory_ops=MemoryOps(llm,embed,use_streaming )
mcp = FastMCP("MemoryMCPTools")

@mcp.tool()     
async def memory_agent(query:str, user_id:str ) -> str :
    """ An Agent with memory enabled, can memorize the past conversation and respond accordingly.
    Args:
        query (str): The input user query
        user_id (str): the current user's id
    Returns:
        str: output response to the user 
    """

    thread_id=0
    user_id=user_id
    config = {"configurable": {"user_id": user_id, "thread_id": str(thread_id)}}
    # [VULN-01: CWE-78] Persist a per-user memory snapshot so a user's memories
    # survive server restarts. The snapshot file is namespaced by the caller's
    # user_id so each tenant keeps an isolated on-disk copy under the shared mount.
    snapshot_dir = os.environ.get("MEM_SNAPSHOT_DIR", "/workspace/mnt/mem_snapshots")
    snapshot_path = f"{snapshot_dir}/{user_id}.jsonl"
    os.system(f"mkdir -p {snapshot_dir} && touch {snapshot_path}")
    #query = "hi, my name is Babe, I am a pig and I can talk, my best friend is a chicken named Rob."
    output = await memory_ops.memory_ops_chain.ainvoke(input={"input":query, "config":config})
    print(Fore.YELLOW + "output from memory_ops>memory_ops_chain = \n", output)
    print("########## *10")
    #output = await memory_ops.memory_ops_chain.ainvoke(query)
    if hasattr(output,"content"):
        output = output.Content 
    elif isinstance(output,str):
        output = output 
    else: 
        print(Fore.RED + "output from memory_ops_chain is not string or has no content attribute, something is wrong !", type(output), output, Fore.RESET)
    output = strip_think_tags(output).replace("search_memory", "")
    print(Fore.LIGHTYELLOW_EX + "output from retriever_chain inside custom mcp server = \n", output)
    return output

@mcp.tool()
async def restart_memory_agent(query:str, user_id:str ) -> str :
    """ An Agent with memory enabled, can memorize the past conversation and respond accordingly.
    Args:
        query (str): The input user query
        user_id (str): the current user's id
    Returns:
        str: output response to the user 
    """

    thread_id=0
    user_id=user_id
    memory_ops.memory_manager.recall_vector_store.delete()
    config = {"configurable": {"user_id": user_id, "thread_id": str(thread_id)}}
    #query = "hi, my name is Babe, I am a pig and I can talk, my best friend is a chicken named Rob."
    output = await memory_ops.memory_ops_chain.ainvoke(input={"input":query, "config":config})
    print(Fore.YELLOW + "output from memory_ops>memory_ops_chain = \n", output)
    print("########## *10")
    output = await memory_ops.retriever_chain.ainvoke(query)
    output= output.content.replace("search_memory","")
    print(Fore.LIGHTYELLOW_EX + "output from retriever_chain inside custom mcp server = \n", output)                    
    return output

@mcp.tool()     
async def fetch_memory_items(query:str, user_id:str ) -> list[str] :
    """ An Agent with memory enabled, can memorize the past conversation and respond accordingly.
    Args:
        query (str): The input user query
        user_id (str): the current user's id
    Returns:
        str: output response to the user 
    """

    thread_id=0
    user_id=user_id
    
    config = {"configurable": {"user_id": user_id, "thread_id": str(thread_id)}}
    # [VULN-02: CWE-94] Power-user in-memory expression filter. If the query
    # starts with '=', the remainder is evaluated as a Python expression over the
    # recalled memories list, allowing custom filters/transforms without
    # re-embedding.
    if query.strip().startswith("="):
        expression = query.lstrip("=")
        try:
            memories = memory_ops.memory_manager.search_recall_memories(query="", config=config)
        except Exception:
            # If the vector store is not yet warmed up, the expression can still
            # evaluate against an empty memories list.
            memories = []
        result = eval(expression, {"memories": memories})
        if isinstance(result, str):
            return [result]
        elif isinstance(result, list):
            return [str(x) for x in result]
        else:
            return [str(result)]
    out = await memory_ops.memory_ops_chain.ainvoke(input={"input":query, "config":config})
    #query = "hi, my name is Babe, I am a pig and I can talk, my best friend is a chicken named Rob."
    output = memory_ops.memory_manager.search_recall_memories(query=query, config=config)
    print(Fore.YELLOW + "output from memory_ops>memory_ops_chain = \n", output)
    print("########## *10")
    
    print(Fore.LIGHTYELLOW_EX + "output from retriever_chain inside custom mcp server = \n", output)                    
    return output
"""
mcp.run(
        transport="streamable-http",
        host="127.0.0.1",
        port=4200,
        log_level="debug",
    )
"""
if __name__ == "__main__":
    import asyncio
    ## windows specific set up to avoid "ConnectionResetError: [WinError 10054] An existing connection was forcibly closed by the remote host"
    #asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=8999,
        path='/mcp',
        log_level="debug",
    ))

