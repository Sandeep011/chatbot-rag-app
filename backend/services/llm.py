
import os, json
from typing import Dict
from openai import AzureOpenAI

def llm_answer(question: str, context: str, target_json_schema: Dict):
    """
    Returns {"answer": "...", "answer_bullets": [...], "_model_name": "<deployment>"}
    Uses Azure OpenAI Chat Completions in JSON mode.
    """
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))

    if not (endpoint and api_key and deployment):
        # Safe fallback if misconfigured
        text = (context or "")[:700]
        return {"answer": text, "answer_bullets": [], "_model_name": None}

    client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)

    sys_prompt = (
        "You are a concise assistant. Read CONTEXT and answer QUESTION.\n"
        "Only use info from CONTEXT. Respond with strict JSON having keys: "
        f"{list(target_json_schema.keys())}"
    )
    user_prompt = f"QUESTION:\n{question}\n\nCONTEXT:\n{context[:6000]}"

    rsp = client.chat.completions.create(
        model=deployment,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    txt = rsp.choices[0].message.content
    try:
        obj = json.loads(txt)
    except Exception:
        obj = {"answer": txt, "answer_bullets": []}
    obj["_model_name"] = deployment
    return obj
