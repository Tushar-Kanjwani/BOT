import pandas as pd
from functools import lru_cache
from together import Together

# --- In-memory storage for chat logs ---
chat_logs = []  # Each entry: {"user_query": ..., "model_response": ..., "total_tokens": ...}

# --- Load players table into an array of dicts ---
def load_players():
    df = pd.read_csv("PCB.csv")
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    return df.to_dict(orient="records")

players = load_players()

@lru_cache(maxsize=1)
def load_context():
    """Loads and formats the PCB.csv into a single text blob."""
    df = pd.DataFrame(players)
    mapping = {
        "Player":"Player","Starting Year":"First Match","Ending Year":"Last Match",
        "Matches Played":"Matches","Innings Batted":"Innings","Not Outs":"Not Outs",
        "Runs Scored":"Runs","Highest Score":"Highest Score","Average Score":"Average",
        "Centuries":"100s","Half-centuries":"50s","Ducks":"Ducks"
    }
    lines = []
    for _, row in df.iterrows():
        parts = []
        for col in df.columns:
            label = mapping.get(col, col)
            parts.append(f"{label}: {row[col]}")
        lines.append(", ".join(parts))
    return "\n".join(lines)

# Initialize Together client & context
client = Together(api_key="tgp_v1_2qPAm5jNcqlwanhcusx0I2Ir7K9ECz8z3vmJFWcsGqM")
context_data = load_context()

def get_cached_response(user_query: str):
    """Returns a previously stored response for this query, if any."""
    for log in reversed(chat_logs):
        if log["user_query"] == user_query:
            return log["model_response"]
    return None

def log_chat(user_query: str, model_response: str, total_tokens: int):
    """Logs a new question/answer into chat_logs."""
    chat_logs.append({
        "user_query": user_query,
        "model_response": model_response,
        "total_tokens": total_tokens
    })

def answer_query(user_query: str) -> tuple[str, int]:
    """
    Returns (answer, total_tokens).  
    Uses caching first; if not found, calls the LLM and logs the result.
    """
    # 1) Try cache
    cached = get_cached_response(user_query)
    if cached:
        return cached, 0

    # 2) Otherwise, ask the model
    resp = client.chat.completions.create(
        model="lgai/exaone-3-5-32b-instruct",
        max_tokens=512,
        messages=[
            {
                "role": "system",
                "content": "You are an assistant who answers questions about cricket players from a dataset."
            },
            {
                "role": "user",
                "content": f"Here is the dataset:\n\n{context_data}\n\nNow answer this question:\n{user_query}"
            }
        ]
    )
    answer = resp.choices[0].message.content
    total = getattr(resp.usage, "total_tokens", 0) if hasattr(resp, "usage") else 0

    # 3) Log and return
    log_chat(user_query, answer, total)
    return answer, total
