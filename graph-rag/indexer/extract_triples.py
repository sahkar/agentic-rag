import openai
import asyncio
import json
from openai import AsyncOpenAI
from dotenv import load_dotenv
load_dotenv()
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

client = AsyncOpenAI()

# currently options are pipeswitch, riscv, vLLM
input_paper = 'riscv'

print(f'Extracting triples for paper: {input_paper}')

# Load clustered chunks
with open(f"clustered_chunks_{input_paper}.json", "r", encoding="utf-8") as f:
    clustered_chunks = json.load(f)

# Flatten all text groups
texts = [chunk for group in clustered_chunks.values() for chunk in group]

# Define prompt format
def build_prompt(text):
    return (
        "Extract factual (subject, predicate, object) triples from the following text.\n"
        "try to make predicate names descriptive.\n"
        "Try to focus on relevant subjects and not be duplicative. Subjects should encapsulate words."
        "Provide all output in upper case.\n"
        "Return in JSON format:\n"
        '[{"subject": ..., "predicate": ..., "object": ...}]\n\n'
        f"Text:\n{text}"
    )

# Async batch call
async def extract_triples_batch(texts, model="gpt-4.1-mini"):
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

    # remove pronouns (other than "we")
    bad_subjects = {"it", "they", "he", "she", "this", "that", "these", "those", "him", "her", "them"}
    triples = [t for t in triples if t["subject"].lower() not in bad_subjects]

    # fuzzy deduplication
    dup = [f"{t['subject']} {t['predicate']} {t['object']}" for t in triples]
    vecs = TfidfVectorizer().fit_transform(dup)
    sim_matrix = cosine_similarity(vecs)

    to_remove = set()
    for i in range(len(sim_matrix)):
        for j in range(i + 1, len(sim_matrix)):
            if sim_matrix[i, j] > 0.95:
                to_remove.add(j)

    deduped = [t for idx, t in enumerate(triples) if idx not in to_remove]

    with open(f"triples_relevant_{input_paper}.json", "w", encoding="utf-8") as f:
        json.dump(deduped, f, indent=2, ensure_ascii=False)
    print(f"Saved triples to triples_relevant_{input_paper}.json")

asyncio.run(main())
