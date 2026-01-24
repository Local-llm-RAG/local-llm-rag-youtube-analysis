from typing import List, Dict

from util.app_settings import AppSettings
from util.loader import load_config

def settings() -> AppSettings:
    return load_config(AppSettings)

def build_simple_chat_prompt(
    history: List[Dict[str, str]],
    user_msg: str,
) -> str:
    lines = [f"### System:\n{settings().llm.system_prompt}\n"]
    for h in history:
        lines.append(f"### User:\n{h['user']}\n")
        lines.append(f"### Assistant:\n{h['assistant']}\n")
    lines.append(f"### User:\n{user_msg}\n### Assistant:\n")
    return "".join(lines)
