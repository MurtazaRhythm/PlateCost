#This file is where we will communicate with our tinyllama model
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "tinyllama"

CATEGORIES = [
    "Produce",
    "Meat",
    "Dairy",
    "Dry Goods",
    "Beverages",
    "Other"
]

def ask_llm(prompt: str) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(OLLAMA_URL, json=payload)
    return response.json()["response"].strip()


def categorize_item(item_name: str) -> str:
    #This prompt is how the ai will determine how to categorize
    prompt = f"""
You are classifying food items from supplier receipts for a restaurant.

Choose ONE category only from the list below:
- Produce
- Meat
- Dairy
- Dry Goods
- Beverages
- Other

Item name: "{item_name}"

Rules:
- Respond with ONLY the category name
- Do NOT explain
- If unsure, respond "Other"
"""
    result = ask_llm(prompt)

    for category in CATEGORIES:
        #so if its one of our main categories put it there if not its in other (.lower is for lower case for better comparisons)
        if category.lower() in result.lower():
            return category

    return "Other"


def generate_insight(category: str, current: float, previous: float) -> str:
    change = current - previous
    percent = (change / previous * 100) if previous > 0 else 0

#start of ai
    prompt = f"""
You are helping a small restaurant owner understand their food spend.

Category: {category}
This week: ${current:.2f}
Last week: ${previous:.2f}
Change: {percent:.1f}%

Write ONE short insight (max 1 sentence).
Plain language. No advice.
"""
#end of ai
    return ask_llm(prompt)
