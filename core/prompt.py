from typing import List, Dict

SYSTEM = "Ти си полезен асистент. Отговаряй на български. Бъди кратък и точен."

def build_simple_chat_prompt(
    history: List[Dict[str, str]],
    user_msg: str,
) -> str:
    lines = [f"### System:\n{SYSTEM}\n"]
    for h in history:
        lines.append(f"### User:\n{h['user']}\n")
        lines.append(f"### Assistant:\n{h['assistant']}\n")
    lines.append(f"### User:\n{user_msg}\n### Assistant:\n")
    return "".join(lines)
