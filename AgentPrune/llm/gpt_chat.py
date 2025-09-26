import os
from typing import List, Union, Optional, Dict, Any

from dotenv import load_dotenv
from tenacity import retry, wait_random_exponential, stop_after_attempt
from openai import OpenAI,AsyncOpenAI

from AgentPrune.llm.format import Message
from AgentPrune.llm.price import cost_count
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
    response = chat_completion.choices[0].message.content
    print(f"[LLM] response={response}",flush=True)
# === 加上 token 输出 ===
    # prompt_text = "".join(
    #     (m.get("content", "") if isinstance(m, dict) else str(m))  # 取 content；没有就空串
    #     if isinstance(m.get("content", ""), str)                   # content 是字符串
    #     else str(m.get("content", ""))                             # 否则转成字符串
    #     for m in msg
    # )
    # price, p_tokens, c_tokens = cost_count(prompt_text, response, model)
    # print(f"[TOKENS] prompt={p_tokens}, completion={c_tokens}, total={p_tokens+c_tokens}, cost={price}")
 
    # return response
    price, p_tokens, c_tokens = cost_count(chat_completion, model)
    print(f"[TOKENS] prompt={p_tokens}, completion={c_tokens}, total={p_tokens + c_tokens}, cost_price={price}", flush=True)

    return response


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

        # 同步客户端（ .env 里的 BASE_URL / API_KEY）
        client = OpenAI(base_url=MINE_BASE_URL, api_key=MINE_API_KEY)
        resp = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
        )
        response = resp.choices[0].message.content
        print(f"[LLM] response={response}",flush=True)
# === 加上 token 输出 ===
        # prompt_text = "".join(
        #     (m.get("content", "") if isinstance(m, dict) else str(m))
        #     if isinstance(m.get("content", ""), str)
        #     else str(m.get("content", ""))
        #     for m in messages
        # )
        # price, p_tokens, c_tokens = cost_count(prompt_text, response, self.model_name)
        # print(f"[TOKENS] prompt={p_tokens}, completion={c_tokens}, total={p_tokens+c_tokens}, cost={price}")
        # return response 
        price, p_tokens, c_tokens = cost_count(resp, self.model_name)
        print(f"[TOKENS] prompt={p_tokens}, completion={c_tokens}, total={p_tokens + c_tokens}, cost_price={price}", flush=True)

        return response

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
