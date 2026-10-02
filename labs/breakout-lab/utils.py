from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from  datetime import datetime
import ast
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings, NVIDIARerank
import os
import json
from typing import List, Literal, Optional
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import get_buffer_string
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage , BaseMessage, ToolMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import uuid
import re
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables import RunnableLambda

from colorama import Fore
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from MemoryManager import MemoryHandler
from colorama import Fore
import random
from dotenv import load_dotenv
import nest_asyncio, asyncio
load_dotenv()
nest_asyncio.apply()


choices=["nvidia/llama-3.3-nemotron-super-49b-v1.5"]

if os.environ["NVIDIA_API_KEY"]:
   print("existing NVIDIA_API_KEY in the environment ", os.environ["NVIDIA_API_KEY"][:7])
else:
    load_dotenv()

if os.environ.get('llm_model') == None:
    llm_model=random.choice(choices)
    print(Fore.LIGHTGREEN_EX +"using llm model =", llm_model )
    
if os.environ.get("embed_model")==None:
    embed_model="nvidia/nv-embedqa-mistral-7b-v2"

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


llm = ChatNVIDIA(model=llm_model)

embed = NVIDIAEmbeddings(model=embed_model,truncate="NONE",)


def check_turns(conv_hist):
    n=len(conv_hist)
    lastNhumanmsg=[msg for msg in conv_hist if type(msg)==HumanMessage ]
    return len(lastNhumanmsg)+1

def conv_items_to_list_of_strs(chat_history):
    ls=[]
    for item in chat_history:
        if isinstance(item, HumanMessage):
            ls.append("Human:"+item.content)
        elif isinstance(item, AIMessage):
            ls.append("AI:"+item.content)
        elif isinstance(item, ToolMessage):
            ls.append("Tool:"+item.content)
        elif isinstance(item, SystemMessage):
            ls.append("System:"+item.content)
        else:
            print("unknow item ", type(item), item)
            pass
    return ls
def fetch_lastN_turns(conv_history , last_N_turns):
    n=len(conv_history)
    lastNhumanmsg=check_turns(conv_history)
    idx=lastNhumanmsg[last_N_turns]
    kept_last_N_turns = conv_history[idx:]
    return kept_last_N_turns

def print_me(inputs):
            
    print(Fore.LIGHTRED_EX + "Inputs: ", inputs,Fore.RESET)
    
    return inputs



## loading memory class 
class MemoryOps:
    """
    Implementing Memory Handler into langchain runnable chain for simplicity
    """  
    def __init__(self, llm: ChatNVIDIA , embed: NVIDIAEmbeddings, use_streaming : bool):
        """
            Initialize the Memory Handler to handle agentic memory operations
        """        
        self.llm = llm
        self.embed=embed
        self.use_streaming = use_streaming
        self.number_of_turns = 3
        self.memory_manager=MemoryHandler(llm,embed,use_streaming )
        self.recall_vector_store = InMemoryVectorStore(self.embed)
        """
        self.retriever = self.recall_vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.5},
        )"""
        self.retriever = self.recall_vector_store.as_retriever(search_kwargs={"k":10})
        
        self.runnable_parallel_1_routing_func = RunnableLambda(self.mem_routing_function)
        self.runnable_parallel_2_create_memory = RunnableLambda(self.create_memory_items)
        
        self.config=None
        
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are assistant with ability to memorize conversations from the user. You should always answer user query based on the following context recalled from your memory:\n<Documents>\n{context}\n</Documents>. \
                    Be polite and helpful.",
                ),
                ("user", "{input}"),
            ]
        )
        
        self.retriever_chain = (
            {"context": self.recall_memory, "input": RunnablePassthrough()}
            | prompt
            | self.llm 
        )
        """
        self.chat_history = ChatMessageHistory()
        self.retriever_with_chat_history = RunnableWithMessageHistory(
            self.retriever_chain,
        lambda session_id: self.chat_history,            
            history_messages_key="chat_history",
        )
        
        memory_retrieve_prompt = 'You are an expert assistant who can keep track of the conversations with the user. \
        you have acess to a memory and am able to recall/retrieve relevent memories relevant to the input user query. \
        If you don't know the answer, just say that you don't know. \
        Summarize to short sentences when you answer to user query.\
        {context}'
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", memory_retrieve_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        
        #contextualize_q_system_prompt = 'Given a chat history and the latest user question \
        #which might reference context in the chat history, formulate a standalone question \
        #which can be understood without the chat history. Do NOT answer the question, \
        #just reformulate it if needed and otherwise return it as is.'
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, contextualize_q_prompt)
        

        self.chat_history_memory_chain =  create_retrieval_chain(history_aware_retriever, question_answer_chain)"""
        
        
        MEM_PROMPT = """You are an expert assistant who can keep track of the conversations with the user. \
        you have acess to a memory and am able to recall/retrieve relevent memories relevant to the input user query. \
        If you don't know the answer, just say that you don't know. \
        Summarize to short sentences when you answer to user query.\
        {context}

        <ChatHistory> : 
        latest chat_history ( up to 3 turns of user-assistant conversation ) : {chat_history}
        conversation summary ( summarized > 3 turns of conversation history ): {chat_history_summarized}
        </ChatHistory> 
        
        current user input query: {input}
        Assistant Response:"""
        self.summary=""
        self.number_of_turns=3
        conv_hist_prompt = ChatPromptTemplate.from_template(MEM_PROMPT)

        self.print_me = RunnableLambda(print_me)
        self.conv_hist_aware_retriever_chain = (
            {"chat_history": itemgetter("chat_history"), "chat_history_summarized": itemgetter("chat_history_summarized"), "context": (self.print_me | itemgetter('input') | self.print_me | self.retriever_chain ), "input": itemgetter("input")}
            | conv_hist_prompt
            | self.llm
            | StrOutputParser()
        )
        self.chat_history = []
        
        self.memory_ops_chain = RunnablePassthrough() | {  # this dict is coerced to a RunnableParallel
        "mem_ops": self.runnable_parallel_1_routing_func,
        "mem_items": self.runnable_parallel_2_create_memory,        
        } | self.execute_memory_operations
    def summarize(self):
        conv_summary_prompt = """You are an expert in summarizing conversation between a user and a bot.
        Your task is to progressively summarize the lines of conversation provided, adding onto the previous summary returning a new summary.

        <EXAMPLE>
        Current summary:
        The human asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is a force for good.
        New lines of conversation:
        Human: Why do you think artificial intelligence is a force for good?
        AI: Because artificial intelligence will help humans reach their full potential.
        New summary:\nThe human asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is a force for good because it will help humans reach their full potential.
        </EXAMPLE>

        Current summary:
        {summary}
        New lines of conversation:{conversations}
        - return ONLY the New Summary and nothing else.
        New summary:"""
        conv_summary_prompt_template = PromptTemplate(        
        template=conv_summary_prompt,
        )
        chat_history_ls=conv_items_to_list_of_strs(self.chat_history)
        summary_chain = (conv_summary_prompt_template | llm | StrOutputParser())
        output= summary_chain.invoke({"summary":self.summary, "conversations":'\n'.join(chat_history_ls)})
        if hasattr(output,"content"):
            output=output.content
        elif isinstance(output,str):
            output=output
        else:
            print(Fore.RED + "output from summarize() ", type(output), output)
        self.summary=output
        #reset chat_history if more than 3 turns
        self.chat_history=[]
        
    async def last_N_conversation_turns(self, ):
        num_turns_thus_far = check_turns(self.chat_history)
        print("num_turns_thus_far", num_turns_thus_far)
        if len(num_turns_thus_far) >= self.number_of_turns :
            self.chat_history = fetch_lastN_turns(self.chat_history, -2)
        
    async def mem_routing_function(self, inputs):
        query=inputs["input"]
        self.memory_manager.current_input=query
        self.config=inputs["config"]
        output=await self.memory_manager.memory_routing(query, self.config)    
        
        return output


    async def create_memory_items(self, inputs):
        query=inputs["input"]
        self.memory_manager.current_input=query
        self.config=inputs["config"]
        memory_items = await self.memory_manager.query_to_memory_items(query=query)
        docs = self.memory_manager.save_recall_memory(memory_items, config=self.config)
        print(Fore.CYAN + "creating memory items =", memory_items, Fore.RESET)
        return docs
    
    async def recall_memory(self, inputs):
        #print(Fore.MAGENTA + "recall memory inputs=\n", inputs, Fore.RESET)
        query=self.memory_manager.current_input
        self.memory_manager.current_input=query
        memory_items = self.memory_manager.search_recall_memories(query, config=self.config)
        print(Fore.MAGENTA + "recall memory items=\n", memory_items, Fore.RESET)
        return memory_items

    async def execute_memory_operations(self,inputs):
        mem_ops=inputs["mem_ops"]
        query=self.memory_manager.current_input
        print(Fore.BLUE +"executing memory operation = ", mem_ops, Fore.RESET)
        turns=check_turns(self.chat_history)
        print(Fore.LIGHTYELLOW_EX + "current turns of conversation =", turns, Fore.RESET)
        if turns > self.number_of_turns:
            self.summarize()
            self.chat_history.append(HumanMessage(content="previous conversation summary:\n"+self.summary + "current user query:\n"+ query))
        else:
            
            self.chat_history.append(HumanMessage(content=query))
        chat_history_ls=conv_items_to_list_of_strs(self.chat_history)
        if 'search_memory' in mem_ops.lower():        
            output = await self.conv_hist_aware_retriever_chain.ainvoke({"chat_history":'\n'.join(chat_history_ls) , "chat_history_summarized": "", "input": query})
            if hasattr(output,"content"):
                output = output.content
            elif isinstance(output,str):
                output=output
            else:
                print(Fore.RED + "output from execute_memory_operation > search memory in mem_ops,lower()", type(output), output)
        
        elif "no operation":
            output=self.llm.invoke(query).content 
        else:
            print("no a valid memory operation> ", mem_ops.lower())
        self.chat_history.append(AIMessage(content=output))
        print("=========="*10 , "\n\n")
        print(Fore.LIGHTRED_EX+"self.chat_hisotyr.messages=\n", len(self.chat_history),  self.chat_history ,Fore.RESET)
        print("\n\n","=========="*10 )
        return output

    
