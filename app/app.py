
################################################################################################################################
print("\nSTARTING TO LOAD LIBRARIES\n")
#import tensorflow
import os
import json
import re
import ast

 
import nest_asyncio
nest_asyncio.apply()
import asyncio
 
from transformers import MistralForCausalLM
from huggingface_hub import snapshot_download


from mistralai import Mistral 

from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer 
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest

from llama_index.core.settings import Settings
from llama_index.llms.mistralai import MistralAI
from llama_index.core import SimpleDirectoryReader, SummaryIndex, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import FunctionTool, ToolMetadata
from llama_index.core.vector_stores import MetadataFilters, FilterCondition
from llama_index.core.objects import ObjectIndex
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner

from llama_index.core.llms import ChatMessage, ChatResponse, MessageRole, CompletionResponse 
 
from llama_index.core.llms.function_calling import FunctionCallingLLM
 
from llama_index.core.agent.workflow.workflow_events import ToolCall



from pydantic import BaseModel, Field
from typing import Any, Optional, List, Sequence
 
 


import importlib
import utilities.utils
importlib.reload(utilities.utils)
from utilities.utils import get_doc_tools


from pathlib import Path

print(f"\nCURRENT WORKING DIRECTORY IS: {os.getcwd()}\n")
#print(os.getcwd())

################################################################################################################################

papers = [
    "metagpt.pdf",
    "longlora.pdf",
    "loftq.pdf",
    "swebench.pdf",
    "selfrag.pdf",
    "zipformer.pdf",
    "values.pdf",
    "finetune_fair_diffusion.pdf",
    "knowledge_card.pdf",
    "metra.pdf",
    "vr_mcl.pdf"
]

################################################################################################################################

hf_token = os.getenv('HUGFACE_AUTH_TOKEN')

if hf_token is None:
    raise ValueError("Hugging Face token not found. Please set it as an environment variable.")
else:
    print("Hugging Face token successfully retrieved.")


model_path = os.path.join(os.getcwd(), 'model_weights')


repo_id = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
snapshot_download(repo_id= repo_id,   
                   allow_patterns=["params.json", # MODEL CONFIGURATION PARAMETERS
                                  "config.json",
                                  "consolidated.safetensors",  
                                  "tokenizer.json", # MODEL TOKENIZER DATA  (vocab, merges, etc.).
                                   "tokenizer_config.json", #  CONFIGURATION INFO LIKE SPECIAL TOKENS AND OTHER SETTINGS 
                                    "tekken.json"
                                   ],  
                   local_dir=model_path,  # Local directory to save the files
                   use_auth_token= hf_token,  # Use authentication token
                   #local_dir_use_symlinks=False  # optional: avoids symlinks, stores real files
                )

print("MISTRAL MODEL WEIGHTS DOWNLOADED")


tokenizer = MistralTokenizer.from_file(f"{model_path}/tekken.json")
mistral_llm = Transformer.from_folder(model_path)

print("MISTRAL MODEL LOADED")

################################################################################################################################

# WRAPPER DEFINITION


# helper functions

def extract_json_blocks(text: str):
    pattern = r"```json\s*(\{.*?\})\s*```"
    matches = re.findall(pattern, text, re.DOTALL)
    return [json.loads(m) for m in matches]

def string_to_object_in_quotes(string_of_python_object):
    try:
        return ast.literal_eval(string_of_python_object)
    except (SyntaxError, ValueError) as e:
        print(f"Error converting string to python object: {e}")
        print(string_of_python_object)
        return string_of_python_object

target_words = r'("choice":|"reason":)'
word2 = "reason"

def split_on_keywords(splitting_words, text):
    result_dict = {}
    parts = re.split(splitting_words, text)
    
    if len(parts) > 1:
        for i in range(1, len(parts), 2):
            key = parts[i].strip()
            value = parts[i+1].strip() if i+1 < len(parts) else ""
            
            if string_to_object_in_quotes(key[0:-1]) == word2:
                result_dict[string_to_object_in_quotes(key[0:-1])] = (
                    string_to_object_in_quotes(value.replace("\n", "").replace("]", "").replace("}", "")
                                               .replace("\\", "!").replace(",", "").replace("{", ""))
                )
            else:
                result_dict[string_to_object_in_quotes(key[0:-1])] = int(value.strip(','))
    else:
        result_dict['text'] = text
    return result_dict

# JSON storage
class JSONStorage:
    def __init__(self, data):
        self.text = data

# Metadata class
class LLMMetadata:
    def __init__(self, model_name, model_type, context_window, version, num_output=1):
        self.model_name = model_name
        self.model_type = model_type
        self.context_window = context_window
        self.version = version
        self.num_output = num_output
        self.is_chat_model = True
        self.is_function_calling_model = True

# MAIN WRAPPER
# If using FunctionCallingLLM, these 3 "METHODS" must be implemented: ".get_tool_calls_from_response()", ".predict_and_call(...)" AND ".predict(...)"
class MistralLlamaIndexWrapper(FunctionCallingLLM, BaseModel):
    tokenizer: Any
    llm: Any
    max_tokens: int = Field(description="Maximum tokens to generate")

    def __init__(self, llm: Any, tokenizer: Any, max_tokens: int, **kwargs):
        super().__init__(llm=llm, tokenizer=tokenizer, max_tokens=max_tokens, **kwargs)
        self.tokenizer = tokenizer
        self.llm = llm
        self.max_tokens = max_tokens

    def _messages_to_prompt(self, messages: List[ChatMessage]) -> str:
        """Convert a list of ChatMessage objects into a single prompt string."""
        prompt = ""
        for msg in messages:
            role = msg.role.value.capitalize()
            prompt += f"{role}: {msg.content}\n"
        return prompt.strip()    
    

    def chat_with_tools(
        self,
        tools: Sequence[Any],
        user_msg: Optional[str],
        chat_history: List[ChatMessage],
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        **kwargs: Any,
    ) -> ChatResponse:
        """Simulate tool-calling chat (stub for now)."""
        # Append user message to chat history
        messages = chat_history + [ChatMessage(role=MessageRole.USER, content=user_msg)]

        # ENSURES "TOOLS" IS A LIST OF DICTIONARIES
        tool_dicts = [{"name": tool.metadata.name, "description": tool.metadata.description} for tool in tools]
          
        return self.chat(messages=messages, tools=tool_dicts, **kwargs)
         

    def get_tool_calls_from_response(self, response: str, functions: Optional[List[ToolMetadata]] = None,
                                     error_on_no_tool_call: bool = False  
                                     ) -> List[ToolCall]:
        # Parse the model's response to extract tool calls
        # Example assumes JSON output with tool name + arguments
        tool_calls = []
        try:
            #print("DEBUG: Response type:", type(response))
            #print("DEBUG: Response content:", response.message.content)
            #parsed = json.loads(response.message.content)
            parsed_blocks = extract_json_blocks(response.message.content)
            for parsed in parsed_blocks:
                for call in parsed.get("tool_calls", []):
                    tool_calls.append(
                        ToolCall(
                            name=call["name"],
                            args=call["arguments"],
                                )
                                        )            
            
        except Exception as e:
            raise ValueError(f"Failed to parse tool calls: {e}")

        return tool_calls
     


    def chat(self, messages: List[ChatMessage], tools: Optional[List[dict]] = None, **kwargs) -> ChatResponse:
        prompt = self._messages_to_prompt(messages)

        # Add tool info to prompt (you could make this prettier for the model)
        if tools:
            tool_list_text = "\n".join([f"{tool['name']}: {tool['description']}" for tool in tools])
            prompt = f"You have access to these tools:\n{tool_list_text}\n\n{prompt}"

        request = ChatCompletionRequest(messages=[UserMessage(content=prompt)])
        tokens = self.tokenizer.encode_chat_completion(request).tokens

        out_tokens, _ = generate(
            [tokens],
            self.llm,
            max_tokens=self.max_tokens,
            temperature=0.0,
            eos_id=self.tokenizer.instruct_tokenizer.tokenizer.eos_id
        )

        result = self.tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])

        #print("\n=== RAW MODEL OUTPUT: START ===\n")
        #print(result)
        #print("\n=== RAW MODEL OUTPUT: END ===\n")
        # Parse for tool usage: match the Function Calling LLM structure
        pattern = re.compile(
            r"Thought:\s*(.+?)\s*Action:\s*(.+?)\s*Action Input:\s*(\{.*?\})",
            re.DOTALL
        )

        matches = pattern.findall(result)

        if matches:
            thought, action, action_input = matches[-1]
            try:
                action_input_json = ast.literal_eval(action_input)
                if not isinstance(action_input_json, dict):
                    raise ValueError("Action input is not a dict")
            except Exception as e:
                print("Failed to parse action input:", action_input)
                # action_input_json = {}
                return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=result.strip()),
                                    tool_calls=[]
                                    )                     
                
            tool_call = ToolCall(
                name=action.strip(),
                args=action_input_json,
            )
            return ChatResponse(
                message=ChatMessage(role=MessageRole.ASSISTANT, content=thought.strip()),
                tool_calls=self.get_tool_calls_from_response(result.strip())
            )
        else:
            return ChatResponse(
                message=ChatMessage(role=MessageRole.ASSISTANT, content=result.strip()),
                tool_calls=[]
            )

    def stream_chat(self, messages: List[ChatMessage], tools: Optional[List[dict]] = None, **kwargs):
        raise NotImplementedError("Streaming not implemented for function calling mode")

    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        # Simple fallback non-function calling completion
        request = ChatCompletionRequest(messages=[UserMessage(content=prompt)])
        tokens = self.tokenizer.encode_chat_completion(request).tokens

        out_tokens, _ = generate(
            [tokens],
            self.llm,
            max_tokens=self.max_tokens,
            temperature=0.0,
            eos_id=self.tokenizer.instruct_tokenizer.tokenizer.eos_id
        )

        result = self.tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])

        return CompletionResponse(
            text=result.strip(),
            raw=result.strip(),
            additional_kwargs={}
        )

    def stream_complete(self, prompt: str, **kwargs):
        for chunk in self.llm.stream(prompt):
            yield chunk

 
    def _prepare_chat_with_tools(self, tools: Optional[List[dict]]) -> str:
        # Stub: Add custom logic if you plan to support tools in structured ways
        return "Tools are not currently supported in a structured way."

    async def achat(self, messages: List[ChatMessage], tools: Optional[List[dict]] = None, **kwargs) -> ChatResponse:
        # Wrap the sync call to chat in a thread-safe async wrapper
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.chat, messages, tools, **kwargs)


    async def acomplete(self, prompt: str, **kwargs) -> CompletionResponse:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.complete, prompt, **kwargs)



    async def astream_chat(self, messages: List[ChatMessage], tools: Optional[List[dict]] = None, **kwargs):
        raise NotImplementedError("Async stream chat not implemented")

    async def astream_complete(self, prompt: str, **kwargs):
        raise NotImplementedError("Async stream complete not implemented")
    

    

    @property
    def metadata(self):
        return LLMMetadata(
            model_name="mistral-small-latest",
            model_type="causal",
            context_window=2048,
            #is_function_calling_model=True,
            version="v0.3"
        )

max_tokens = 2000
mistral_llm_wrapper = MistralLlamaIndexWrapper(llm=mistral_llm, tokenizer=tokenizer, max_tokens=max_tokens)

print("LLM MODEL WRAPPED !!!!!!!!")


###############################################################################################################################

def create_tools_by_processing_documents(docs_path, docs_list, embedding_model, llm, similarity_top_k):

    tool_pairs_dict = {}

    for i, doc in enumerate(docs_list):

        file_path = os.path.join(docs_path, docs_list[i])
        name = Path(file_path).stem      
        vector_tool, summary_tool = get_doc_tools(file_path, name, pre_trained_model = embedding_model, llm = llm, similarity_top_k = similarity_top_k)
         

        tool_pairs_dict[name] = [vector_tool, summary_tool]

    
    keys_list = list(tool_pairs_dict.keys())
    all_tools = [t for name in keys_list for t in tool_pairs_dict[name]]
 
    print(f"ALL DOCUMENTS EMBEDDED!! \n{len(all_tools)} TOOLS CREATED")
  

    return tool_pairs_dict, all_tools

 
datasets_storage_on_disk = os.path.join(os.getcwd(), 'llamaindex_datasets') 

tool_pairs_dict, all_tools_flat_list = create_tools_by_processing_documents(docs_path = datasets_storage_on_disk, 
                                     #docs_list = papers, 
                                     docs_list = ["metagpt.pdf", "swebench.pdf"], 
                                     embedding_model = "sentence-transformers/all-MiniLM-L6-v2",
                                     llm = mistral_llm_wrapper, 
                                     similarity_top_k = 2)


print("TOOLS CREATED")

##############################################################################################################################

# CREATE AGENT

def agent_with_tools_in_vector_store(tools_list, vector_store, embed_model, similarity_top_k, llm):

    
    print("\nSTARTS: TOOLS SERIALIZATION AND EMBEDDING THEM AS VECTORS ")
    obj_index = ObjectIndex.from_objects(tools_list, index_cls=vector_store, embed_model=embed_model)
    print("\nENDS: TOOLS SERIALIZATION AND EMBEDDING THEM AS VECTORS ")

    
    print("\nSTARTS: DEFINE A RETRIEVER OVER THE EMBEDDINGS OF THE TOOLS")
    obj_retriever = obj_index.as_retriever(similarity_top_k=similarity_top_k)
    print("\nENDS: DEFINE A RETRIEVER OVER THE EMBEDDINGS OF THE TOOLS")



    # When relevant, select the most appropriate tool and provide a clear explanation of your reasoning.
    # If a tool is needed, think through your approach first, then specify the tool and its arguments.   
    agent_worker = FunctionCallingAgentWorker.from_tools(tool_retriever=obj_retriever, llm=llm, verbose=True,
                                                          system_prompt=""" 
                                                                        You are an agent designed to answer queries over a set of given papers.
                                                                        You can use the tools provided to either retrieve specific information from the documents or 
                                                                        provide summaries.                                                              
                                                                        Please always use the tools provided to answer a question, and do not rely on prior knowledge.
                                                                        ALWAYS PROVIDE CLEARLY THE FINAL ANSWER TO THE QUERY THAT YOU HAVE REACHED
                                                                        """
                                                        )
    

    agent = AgentRunner(agent_worker)
    print("\nAGENT CREATED: ")

    return agent

print("START AGENT REASONING")



embedding_model = HuggingFaceEmbedding("sentence-transformers/all-MiniLM-L6-v2") 

query= "Tell me about the evaluation dataset used in MetaGPT and compare it against SWE-Bench"

response = agent_with_tools_in_vector_store(tools_list=all_tools_flat_list, 
                                            vector_store=VectorStoreIndex, 
                                            embed_model = embedding_model,
                                            similarity_top_k=3,
                                            llm=mistral_llm_wrapper).query(query)


print()
print("\nRESPONSE TO THE QUERY IS: ", str(response))
