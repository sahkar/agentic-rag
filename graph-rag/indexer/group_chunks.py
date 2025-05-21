from collections import defaultdict
import json
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import nltk
import fitz 

nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(path):
    doc = fitz.open(path)
    text = "\n".join([page.get_text() for page in doc])
    doc.close()
    return text

# PDF to text
text = extract_text_from_pdf("input_pdf/pipeswitch.pdf")

# Basic regex clean
text = re.sub(r'\s+', ' ', text)
text = re.sub(r'[^a-zA-Z0-9.,;:!?()\'\" -]', '', text)

# Sentence chunking
chunks = sent_tokenize(text)
chunks = [c.strip() for c in chunks if c.strip()]

# Embeddings
embeddings = model.encode(chunks)

# Similarity matrix
sim_matrix = cosine_similarity(embeddings)

# Optional: print most similar pairs (cosine > 0.8)
threshold = 0.8
for i in range(len(chunks)):
    for j in range(i+1, len(chunks)):
        if sim_matrix[i][j] > threshold:
            print(f"[{i}] {chunks[i]}")
            print(f"[{j}] {chunks[j]}")
            print(f"Similarity: {sim_matrix[i][j]:.2f}\n")

# Clustering similar chunks (optional)
clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.3, metric='cosine', linkage='average')
labels = clustering.fit_predict(embeddings)

# After getting 'labels' from AgglomerativeClustering
clustered_chunks = defaultdict(list)

for label, chunk in zip(labels, chunks):
    clustered_chunks[label].append(chunk)

# Now you can print them
for cluster_id, group in clustered_chunks.items():
    print(f"Group {cluster_id}:")
    for chunk in group:
        print(chunk[:200])  # Preview first 200 chars
    print("\n---\n")

serializable_clusters = {str(cluster_id): group for cluster_id, group in clustered_chunks.items()}

with open("clustered_chunks.json", "w", encoding="utf-8") as f:
    json.dump(serializable_clusters, f, indent=2, ensure_ascii=False)

print("Clusters saved to clustered_chunks.json")
