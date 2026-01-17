from typing import Optional, List, Dict
from llama_cpp import Llama

llm: Optional[Llama] = None
model_path: Optional[str] = None

# in-memory chat history
history: List[Dict[str, str]] = []
