import openai
import asyncio
import json
from openai import AsyncOpenAI
from dotenv import load_dotenv
load_dotenv()

client = AsyncOpenAI()

# Load clustered chunks
with open("clustered_chunks.json", "r", encoding="utf-8") as f:
    clustered_chunks = json.load(f)

# Flatten all text groups
texts = [chunk for group in clustered_chunks.values() for chunk in group]

# Define your prompt format
def build_prompt(text):
    return (
        "Extract factual (subject, predicate, object) triples from the following text.\n"
        "Return in JSON format:\n"
        '[{"subject": ..., "predicate": ..., "object": ...}]\n\n'
        f"Text:\n{text}"
    )

# Async batch call
async def extract_triples_batch(texts, model="gpt-3.5-turbo"):
    results = []
    for i in range(0, len(texts), 10):  # batch size 10
        batch = texts[i:i+10]
        tasks = [
            client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": build_prompt(text)},
                ],
                temperature=0
            )
            for text in batch
        ]
        responses = await asyncio.gather(*tasks)
        for resp in responses:
            try:
                result = json.loads(resp.choices[0].message.content)
                results.extend(result)
            except Exception:
                continue
    return results

# Run extraction and save results
async def main():
    triples = await extract_triples_batch(texts)
    with open("triples.json", "w", encoding="utf-8") as f:
        json.dump(triples, f, indent=2, ensure_ascii=False)
    print("Saved triples to triples.json")

asyncio.run(main())
