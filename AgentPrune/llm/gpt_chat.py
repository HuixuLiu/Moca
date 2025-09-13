# import aiohttp
# from typing import List, Union, Optional
# from tenacity import retry, wait_random_exponential, stop_after_attempt
# from typing import Dict, Any
# from dotenv import load_dotenv
# import os
# from openai import OpenAI, AsyncOpenAI

# from AgentPrune.llm.format import Message
# from AgentPrune.llm.price import cost_count
# from AgentPrune.llm.llm import LLM
# from AgentPrune.llm.llm_registry import LLMRegistry


# OPENAI_API_KEYS = ['']
# BASE_URL = ''

# load_dotenv()
# MINE_BASE_URL = os.getenv('BASE_URL')
# MINE_API_KEY = os.getenv('API_KEY')


# @retry(wait=wait_random_exponential(max=300), stop=stop_after_attempt(3))
# async def achat(
#     model: str,
#     msg: List[Dict],):
#     client = AsyncOpenAI(base_url = MINE_BASE_URL, api_key = MINE_API_KEY,)
#     chat_completion = await client.chat.completions.create(messages = msg,model = model,)
#     response = chat_completion.choices[0].message.content
#     return response
    

# @LLMRegistry.register('GPTChat')
# class GPTChat(LLM):

#     def __init__(self, model_name: str):
#         self.model_name = model_name

#     async def agen(
#         self,
#         messages: List[Message],
#         max_tokens: Optional[int] = None,
#         temperature: Optional[float] = None,
#         num_comps: Optional[int] = None,
#         ) -> Union[List[str], str]:

#         if max_tokens is None:
#             max_tokens = self.DEFAULT_MAX_TOKENS
#         if temperature is None:
#             temperature = self.DEFAULT_TEMPERATURE
#         if num_comps is None:
#             num_comps = self.DEFUALT_NUM_COMPLETIONS
        
#         if isinstance(messages, str):
#             messages = [{'role':'user', 'content':'messages'}]
#         return await achat(self.model_name,messages)
    
#     def gen(
#         self,
#         messages: List[Message],
#         max_tokens: Optional[int] = None,
#         temperature: Optional[float] = None,
#         num_comps: Optional[int] = None,
#     ) -> Union[List[str], str]:
#         pass

import os
from typing import List, Union, Optional, Dict, Any

from dotenv import load_dotenv
from tenacity import retry, wait_random_exponential, stop_after_attempt
from openai import OpenAI,AsyncOpenAI

from AgentPrune.llm.format import Message
from AgentPrune.llm.llm import LLM
from AgentPrune.llm.llm_registry import LLMRegistry

# 读取 .env（允许覆盖当前环境变量，避免旧值残留）
load_dotenv(override=True)

# 兼容多种环境变量命名（Ollama/OpenAI 皆可）
MINE_BASE_URL = (
    os.getenv("BASE_URL")
    or os.getenv("OPENAI_BASE_URL")
    or "http://127.0.0.1:11434/v1"   # 本地 Ollama 的 OpenAI 兼容端默认
)
MINE_API_KEY = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY") or "ollama"  # 非空即可

@retry(wait=wait_random_exponential(max=300), stop=stop_after_attempt(3))
async def achat(model: str, msg: List[Dict[str, Any]]):
    client = AsyncOpenAI(base_url=MINE_BASE_URL, api_key=MINE_API_KEY)
    # 调试：打印实际使用的模型与端点
    print(f"[LLM] base_url={MINE_BASE_URL} model={model}", flush=True)
    chat_completion = await client.chat.completions.create(
        model=model,
        messages=msg,
    )
    return chat_completion.choices[0].message.content


@LLMRegistry.register("GPTChat")
class GPTChat(LLM):
    def __init__(self, model_name: Optional[str]):
        # 总是优先使用环境变量 MODEL；没有的话再用传入；最后兜底到本地 ollama 模型
        self.model_name = os.getenv("MODEL") or model_name or "llama3.1:8b-instruct-fp16"

    def gen(
        self,
        messages: List[Message] | str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        # 与 agen 一样的默认值
        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        # 支持传入纯字符串
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        # 同步客户端（走你 .env 里的 BASE_URL / API_KEY）
        client = OpenAI(base_url=MINE_BASE_URL, api_key=MINE_API_KEY)
        resp = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
        )
        return resp.choices[0].message.content    
    async def agen(
        self,
        messages: List[Message] | str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        return await achat(self.model_name, messages)
