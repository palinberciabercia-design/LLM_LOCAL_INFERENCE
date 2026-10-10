
import pandas as pd
import ollama
import os

os.makedirs("dataset", exist_ok=True)

prompts = pd.read_csv("dataset/prompts.csv")
prompts.columns = prompts.columns.str.strip().str.upper()

# Find the correct column names
prompt_id_col = "PROMPT_ID"
prompt_col = "PROMPT"

if "DOMAIN" in prompts.columns:
    domain_col = "DOMAIN"
elif "DOMAINE" in prompts.columns:
    domain_col = "DOMAINE"
else:
    domain_col = None

model = "llama3.2"
responses = []
comparison = []
history = []

for _, row in prompts.iterrows():
    prompt = str(row[prompt_col])
    prompt_id = row[prompt_id_col]
    domain = row[domain_col] if domain_col else ""

    print("\nUser:", prompt)

    result = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

    answer = result["message"]["content"]
    print("Assistant:", answer)

    word_count = len(answer.split())
    char_count = len(answer)

    responses.append({
        "PROMPT_ID": prompt_id,
        "DOMAIN": domain,
        "PROMPT": prompt,
        "RESPONSE": answer
    })

    comparison.append({
        "PROMPT_ID": prompt_id,
        "DOMAIN": domain,
        "PROMPT": prompt,
        "RESPONSE": answer,
        "WORD_COUNT": word_count,
        "CHARACTER_COUNT": char_count
    })

    history.append(f"PROMPT_ID: {prompt_id}\n")
    history.append(f"DOMAIN: {domain}\n")
    history.append(f"PROMPT: {prompt}\n")
    history.append(f"RESPONSE: {answer}\n")
    history.append("-" * 50 + "\n")

pd.DataFrame(responses).to_csv(
    "dataset/llm_responses.csv", index=False
)

pd.DataFrame(comparison).to_excel(
    "dataset/response_comparison.xlsx", index=False
)

with open(
    "dataset/conversation_history.txt",
    "w",
    encoding="utf-8"
) as file:
    file.writelines(history)

print("\nAll output files saved successfully!")
print("llm_responses.csv created")
print("response_comparison.xlsx created")
print("conversation_history.txt created")
