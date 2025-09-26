# from AgentPrune.utils.globals import Cost, PromptTokens, CompletionTokens
# import tiktoken
# # GPT-4:  https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo
# # GPT3.5: https://platform.openai.com/docs/models/gpt-3-5
# # DALL-E: https://openai.com/pricing

# # def cal_token(model:str, text:str):
# #     encoder = tiktoken.encoding_for_model(model)
# #     num_tokens = len(encoder.encode(text))
# #     return num_tokens

# def cal_token(model: str, text: str):
#     try:
#         encoder = tiktoken.encoding_for_model(model)
#     except KeyError:
#         # 本地模型或未知模型时，用通用编码
#         encoder = tiktoken.get_encoding("cl100k_base")
#     num_tokens = len(encoder.encode(text or ""))
#     return num_tokens

# def cost_count(prompt, response, model_name):
#     branch: str
#     prompt_len: int
#     completion_len: int
#     price: float

#     prompt_len = cal_token(model_name, prompt)
#     completion_len = cal_token(model_name, response)
#     if "gpt-4" in model_name:
#         branch = "gpt-4"
#         price = prompt_len * OPENAI_MODEL_INFO[branch][model_name]["input"] /1000 + \
#                 completion_len * OPENAI_MODEL_INFO[branch][model_name]["output"] /1000
#     elif "gpt-3.5" in model_name:
#         branch = "gpt-3.5"
#         price = prompt_len * OPENAI_MODEL_INFO[branch][model_name]["input"] /1000 + \
#             completion_len * OPENAI_MODEL_INFO[branch][model_name]["output"] /1000
#     elif "dall-e" in model_name:
#         branch = "dall-e"
#         price = 0.0
#         prompt_len = 0
#         completion_len = 0
#     else:
#         branch = "other"
#         price = 0.0

#     Cost.instance().value += price
#     PromptTokens.instance().value += prompt_len
#     CompletionTokens.instance().value += completion_len

#     # print(f"Prompt Tokens: {prompt_len}, Completion Tokens: {completion_len}")
#     return price, prompt_len, completion_len

# OPENAI_MODEL_INFO ={
#     "gpt-4": {
#         "current_recommended": "gpt-4-1106-preview",
#         "gpt-4-0125-preview": {
#             "context window": 128000, 
#             "training": "Jan 2024", 
#             "input": 0.01, 
#             "output": 0.03
#         },      
#         "gpt-4-1106-preview": {
#             "context window": 128000, 
#             "training": "Apr 2023", 
#             "input": 0.01, 
#             "output": 0.03
#         },
#         "gpt-4-vision-preview": {
#             "context window": 128000, 
#             "training": "Apr 2023", 
#             "input": 0.01, 
#             "output": 0.03
#         },
#         "gpt-4": {
#             "context window": 8192, 
#             "training": "Sep 2021", 
#             "input": 0.03, 
#             "output": 0.06
#         },
#         "gpt-4-0314": {
#             "context window": 8192, 
#             "training": "Sep 2021", 
#             "input": 0.03, 
#             "output": 0.06
#         },
#         "gpt-4-32k": {
#             "context window": 32768, 
#             "training": "Sep 2021", 
#             "input": 0.06, 
#             "output": 0.12
#         },
#         "gpt-4-32k-0314": {
#             "context window": 32768, 
#             "training": "Sep 2021", 
#             "input": 0.06, 
#             "output": 0.12
#         },
#         "gpt-4-0613": {
#             "context window": 8192, 
#             "training": "Sep 2021", 
#             "input": 0.06, 
#             "output": 0.12
#         },
#         "gpt-4o": {
#             "context window": 128000, 
#             "training": "Jan 2024", 
#             "input": 0.005, 
#             "output": 0.015
#         }, 
#     },
#     "gpt-3.5": {
#         "current_recommended": "gpt-3.5-turbo-1106",
#         "gpt-3.5-turbo-0125": {
#             "context window": 16385, 
#             "training": "Jan 2024", 
#             "input": 0.0010, 
#             "output": 0.0020
#         },
#         "gpt-3.5-turbo-1106": {
#             "context window": 16385, 
#             "training": "Sep 2021", 
#             "input": 0.0010, 
#             "output": 0.0020
#         },
#         "gpt-3.5-turbo-instruct": {
#             "context window": 4096, 
#             "training": "Sep 2021", 
#             "input": 0.0015, 
#             "output": 0.0020
#         },
#         "gpt-3.5-turbo": {
#             "context window": 4096, 
#             "training": "Sep 2021", 
#             "input": 0.0015, 
#             "output": 0.0020
#         },
#         "gpt-3.5-turbo-0301": {
#             "context window": 4096, 
#             "training": "Sep 2021", 
#             "input": 0.0015, 
#             "output": 0.0020
#         },
#         "gpt-3.5-turbo-0613": {
#             "context window": 16384, 
#             "training": "Sep 2021", 
#             "input": 0.0015, 
#             "output": 0.0020
#         },
#         "gpt-3.5-turbo-16k-0613": {
#             "context window": 16384, 
#             "training": "Sep 2021", 
#             "input": 0.0015, 
#             "output": 0.0020
#         }
#     },
#     "dall-e": {
#         "current_recommended": "dall-e-3",
#         "dall-e-3": {
#             "release": "Nov 2023",
#             "standard": {
#                 "1024×1024": 0.040,
#                 "1024×1792": 0.080,
#                 "1792×1024": 0.080
#             },
#             "hd": {
#                 "1024×1024": 0.080,
#                 "1024×1792": 0.120,
#                 "1792×1024": 0.120
#             }
#         },
#         "dall-e-2": {
#             "release": "Nov 2022",
#             "1024×1024": 0.020,
#             "512×512": 0.018,
#             "256×256": 0.016
#         }
#     }
# }

# price.py
from AgentPrune.utils.globals import Cost, PromptTokens, CompletionTokens

# OpenAI 价格信息（保留原有表）
OPENAI_MODEL_INFO = {
    "gpt-4": {
        "current_recommended": "gpt-4-1106-preview",
        "gpt-4-0125-preview": {"context window": 128000, "training": "Jan 2024", "input": 0.01, "output": 0.03},
        "gpt-4-1106-preview": {"context window": 128000, "training": "Apr 2023", "input": 0.01, "output": 0.03},
        "gpt-4-vision-preview": {"context window": 128000, "training": "Apr 2023", "input": 0.01, "output": 0.03},
        "gpt-4": {"context window": 8192, "training": "Sep 2021", "input": 0.03, "output": 0.06},
        "gpt-4-0314": {"context window": 8192, "training": "Sep 2021", "input": 0.03, "output": 0.06},
        "gpt-4-32k": {"context window": 32768, "training": "Sep 2021", "input": 0.06, "output": 0.12},
        "gpt-4-32k-0314": {"context window": 32768, "training": "Sep 2021", "input": 0.06, "output": 0.12},
        "gpt-4-0613": {"context window": 8192, "training": "Sep 2021", "input": 0.06, "output": 0.12},
        "gpt-4o": {"context window": 128000, "training": "Jan 2024", "input": 0.005, "output": 0.015},
    },
    "gpt-3.5": {
        "current_recommended": "gpt-3.5-turbo-1106",
        "gpt-3.5-turbo-0125": {"context window": 16385, "training": "Jan 2024", "input": 0.0010, "output": 0.0020},
        "gpt-3.5-turbo-1106": {"context window": 16385, "training": "Sep 2021", "input": 0.0010, "output": 0.0020},
        "gpt-3.5-turbo-instruct": {"context window": 4096, "training": "Sep 2021", "input": 0.0015, "output": 0.0020},
        "gpt-3.5-turbo": {"context window": 4096, "training": "Sep 2021", "input": 0.0015, "output": 0.0020},
        "gpt-3.5-turbo-0301": {"context window": 4096, "training": "Sep 2021", "input": 0.0015, "output": 0.0020},
        "gpt-3.5-turbo-0613": {"context window": 16384, "training": "Sep 2021", "input": 0.0015, "output": 0.0020},
        "gpt-3.5-turbo-16k-0613": {"context window": 16384, "training": "Sep 2021", "input": 0.0015, "output": 0.0020},
    },
    "dall-e": {
        "current_recommended": "dall-e-3",
        "dall-e-3": {
            "release": "Nov 2023",
            "standard": {"1024×1024": 0.040, "1024×1792": 0.080, "1792×1024": 0.080},
            "hd": {"1024×1024": 0.080, "1024×1792": 0.120, "1792×1024": 0.120},
        },
        "dall-e-2": {"release": "Nov 2022", "1024×1024": 0.020, "512×512": 0.018, "256×256": 0.016},
    },
}


def _get(obj, path, default=None):
    """
    安全获取属性/键：支持 OpenAI SDK 的对象属性和 dict。
    path 形如 "usage.prompt_tokens"
    """
    cur = obj
    for key in path.split("."):
        if cur is None:
            return default
        # 先尝试属性
        if hasattr(cur, key):
            cur = getattr(cur, key)
            continue
        # 再尝试映射/字典
        if isinstance(cur, dict):
            cur = cur.get(key, default)
            continue
        # OpenAI SDK 里有些字段是内部 dict/list
        try:
            cur = cur[key]  # 尝试 __getitem__
        except Exception:
            return default
    return cur if cur is not None else default


def _extract_usage(resp):
    """
    从响应对象里抽 usage。兼容：
    - OpenAI：resp.usage.prompt_tokens / completion_tokens
    - 一些 Ollama 的 OpenAI 兼容层也会返回 usage
    """
    prompt_tokens = _get(resp, "usage.prompt_tokens", 0)
    completion_tokens = _get(resp, "usage.completion_tokens", 0)

    # 如果 usage 不存在，但 total tokens 有值，尽量不瞎算，保持 0
    if not isinstance(prompt_tokens, int):
        prompt_tokens = 0
    if not isinstance(completion_tokens, int):
        completion_tokens = 0

    return prompt_tokens, completion_tokens


def _calc_openai_price(model_name: str, p_tokens: int, c_tokens: int) -> float:
    """
    只有 OpenAI 官方模型才计价；其余（如本地 Ollama）返回 0。
    """
    price = 0.0
    branch = None
    if "gpt-4" in model_name:
        branch = "gpt-4"
    elif "gpt-3.5" in model_name:
        branch = "gpt-3.5"
    elif "dall-e" in model_name:
        # 图像这里保持 0，且 tokens = 0
        return 0.0

    if branch and model_name in OPENAI_MODEL_INFO.get(branch, {}):
        info = OPENAI_MODEL_INFO[branch][model_name]
        price = (p_tokens * info["input"] + c_tokens * info["output"]) / 1000.0
    return price


def cost_count(resp, model_name: str):
    """
    直接从 LLM 响应对象里读取 usage 统计，不做本地分词，不阻塞网络。
    返回：price, prompt_tokens, completion_tokens
    """
    p_tokens, c_tokens = _extract_usage(resp)

    # 价格：OpenAI 模型按表计价；Ollama/其他一律 0
    price = _calc_openai_price(model_name, p_tokens, c_tokens)

    # 全局累加
    Cost.instance().value += price
    PromptTokens.instance().value += p_tokens
    CompletionTokens.instance().value += c_tokens

    return price, p_tokens, c_tokens


