# functions.py
# Function Calling Helper Functions (vendored for self-contained homework2 submission)
# Original course material: Tim Fraser

import json
import sys
import time

import pandas as pd
import requests

DEFAULT_MODEL = "smollm2:1.7b"
PORT = 11434
OLLAMA_HOST = f"http://localhost:{PORT}"
CHAT_URL = f"{OLLAMA_HOST}/api/chat"
REQUEST_TIMEOUT = 300
OLLAMA_TAGS_URL = f"{OLLAMA_HOST}/api/tags"


def ensure_ollama_available(max_wait_seconds: int = 15, poll_interval_seconds: float = 0.5) -> None:
    deadline = time.time() + max_wait_seconds
    last_err = None
    while time.time() < deadline:
        try:
            r = requests.get(OLLAMA_TAGS_URL, timeout=5)
            if r.ok:
                return
        except Exception as e:
            last_err = e
        time.sleep(poll_interval_seconds)

    raise RuntimeError(
        "Ollama is not reachable at localhost:11434. Start it with `ollama serve` or run "
        "`python 01_ollama.py` from the repository root.\n"
        f"Last error: {last_err}"
    )


def agent(messages, model=DEFAULT_MODEL, output="text", tools=None, all=False):
    if tools is None:
        ensure_ollama_available()
        body = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": 500},
        }
        response = requests.post(CHAT_URL, json=body, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        result = response.json()
        return result["message"]["content"]
    else:
        ensure_ollama_available()
        body = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "stream": False,
            "options": {"num_predict": 500},
        }
        response = requests.post(CHAT_URL, json=body, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        result = response.json()

        if "tool_calls" in result.get("message", {}):
            tool_calls = result["message"]["tool_calls"]
            for tool_call in tool_calls:
                func_name = tool_call["function"]["name"]
                raw_args = tool_call["function"].get("arguments", {})
                func_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                func = globals().get(func_name)
                if func is None:
                    for depth in range(1, 6):
                        try:
                            frame = sys._getframe(depth)
                            func = frame.f_globals.get(func_name)
                            if func is not None:
                                break
                        except ValueError:
                            break
                if func:
                    tool_output = func(**func_args)
                    tool_call["output"] = tool_output

        if all:
            return result
        else:
            if "tool_calls" in result.get("message", {}):
                if output == "tools":
                    return tool_calls
                else:
                    return tool_calls[-1].get("output", result["message"]["content"])
            return result["message"]["content"]


def agent_run(role, task, tools=None, output="text", model=DEFAULT_MODEL):
    messages = [
        {"role": "system", "content": role},
        {"role": "user", "content": task},
    ]
    return agent(messages=messages, model=model, output=output, tools=tools)


def df_as_text(df):
    tab = df.to_markdown(index=False)
    return tab
