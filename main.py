import os
import git
import faiss
import json
import re
import pickle
import numpy as np
import requests
from typing import List, Dict
import streamlit as st
from openai import AzureOpenAI

# --- Setup Azure OpenAI Client ---
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

DEPLOYMENT_CHAT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
DEPLOYMENT_EMBED = os.getenv("AZURE_OPENAI_DEPLOYMENT_EMBED")

# --- JIRA Settings (Optional) ---
ENABLE_JIRA_FETCH = True
JIRA_BASE_URL = os.getenv("JIRA_BASE_URL")
JIRA_AUTH = (os.getenv("JIRA_USER_EMAIL"), os.getenv("JIRA_API_TOKEN"))

# --- Setup Repo ---
REPO_PATH = "path"
repo = git.Repo(REPO_PATH)

NULL_TREE = '4b825dc642cb6eb9a060e54bf8d69288fbee4904'

CACHE_FILE = "commit_embedding_cache.pkl"

def fetch_commits(max_count=500) -> List[Dict]:
    commits = []
    for commit in repo.iter_commits('main', max_count=max_count):
        diff_data = commit.diff(commit.parents or NULL_TREE, create_patch=True)
        full_diff = "\n".join(p.diff.decode('utf-8', errors='ignore') for p in diff_data if hasattr(p, 'diff'))
        commits.append({
            "hash": commit.hexsha,
            "author": commit.author.name,
            "date": commit.committed_datetime.isoformat(),
            "message": commit.message.strip(),
            "diff": full_diff,
            "files": list(commit.stats.files.keys()),
            "is_merge": len(commit.parents) > 1
        })
    return commits

def extract_jira_ids(message: str) -> List[str]:
    return re.findall(r"[A-Z]+-\d+", message)

def fetch_jira_description(jira_id: str) -> str:
    if not ENABLE_JIRA_FETCH or not jira_id:
        return ""
    try:
        url = f"{JIRA_BASE_URL}{jira_id}"
        response = requests.get(url, auth=JIRA_AUTH)
        if response.status_code == 200:
            return f"JIRA Details for {jira_id} fetched successfully."
        else:
            return f"[Could not fetch JIRA {jira_id}]"
    except Exception as e:
        return f"[Error fetching JIRA {jira_id}: {e}]"

def classify_commit_llm(message: str) -> str:
    prompt = f"""
Classify the following commit message into one of the categories: Feature, Bugfix, Refactor, Documentation, Test, Other.
Commit Message: {message}
Just respond with one word.
"""
    response = client.chat.completions.create(
        model=DEPLOYMENT_CHAT,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip().lower()

def chunk_and_summarize(commit: Dict) -> str:
    diff_text = commit['diff']
    lines = diff_text.splitlines()
    chunks = []
    current_chunk = ""

    for line in lines:
        if len(current_chunk) + len(line) < 1000:
            current_chunk += line + "\n"
        else:
            chunks.append(current_chunk)
            current_chunk = line + "\n"

    if current_chunk:
        chunks.append(current_chunk)

    summaries = []

    for chunk in chunks:
        prompt = f"""
Summarize this code change in simple technical English:
{chunk}
"""
        response = client.chat.completions.create(
            model=DEPLOYMENT_CHAT,
            messages=[{"role": "user", "content": prompt}]
        )
        summaries.append(response.choices[0].message.content.strip())

    return " ".join(summaries)

def embed_text_azure(texts: List[str]) -> np.ndarray:
    response = client.embeddings.create(
        model=DEPLOYMENT_EMBED,
        input=texts
    )
    embeddings = [d.embedding for d in response.data]
    return np.array(embeddings)

def build_vector_db(commits: List[Dict]):
    if os.path.exists(CACHE_FILE):
        print("Loading cached embeddings...")
        with open(CACHE_FILE, "rb") as f:
            index, meta = pickle.load(f)
        return index, meta

    summaries = []
    meta = []

    for commit in commits:
        commit['type'] = classify_commit_llm(commit['message'])
        jira_ids = extract_jira_ids(commit['message'])
        jira_details = " ".join([fetch_jira_description(jid) for jid in jira_ids])
        commit['jira_ids'] = jira_ids
        commit['jira_details'] = jira_details
        commit['summary'] = chunk_and_summarize(commit)
        summaries.append(commit['summary'] + " " + jira_details)
        meta.append(commit)

    embeddings = embed_text_azure(summaries)

    index = faiss.IndexFlatL2(1536)
    index.add(np.array(embeddings))

    with open(CACHE_FILE, "wb") as f:
        pickle.dump((index, meta), f)

    return index, meta

def search_commits(index, meta, query, top_k=5):
    query_embed = embed_text_azure([query])
    distances, indices = index.search(np.array(query_embed), top_k)

    results = []
    for idx in indices[0]:
        if idx < len(meta):
            results.append(meta[idx])
    return results

def answer_query(index, meta, question):
    related_commits = search_commits(index, meta, question)
    context = "\n".join([
        f"Commit Hash: {c['hash']}\nAuthor: {c['author']}\nDate: {c['date']}\nIs Merge: {c['is_merge']}\nMessage: {c['message']}\nJIRA IDs: {','.join(c.get('jira_ids', []))}\nJIRA Details: {c.get('jira_details', '')}\nSummary: {c['summary']}\n"
        for c in related_commits
    ])

    prompt = f"""
You are a senior engineer. Based on the following commit summaries, metadata, and JIRA details, answer the question. Also reference relevant commit hashes when useful.
Context:
{context}

Question: {question}
Answer:
"""
    print("waiting on response")
    response = client.chat.completions.create(
        model=DEPLOYMENT_CHAT,
        messages=[{"role": "user", "content": prompt}]
    )
    print("got response from llm:" , response)
    return response.choices[0].message.content.strip()

def run_streamlit_app(index, metadata):
    st.title("🔎 Commit History AI Assistant (with JIRA Integration + Cached Embeddings)")

    if st.button("Clear Cache and Reload"):
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
            st.success("Cache cleared! Please restart the app.")

    query = st.text_input("Ask a question about your codebase:")

    if query:
        with st.spinner("Searching commits and thinking..."):
            answer = answer_query(index, metadata, query)
            st.success(answer)

if __name__ == "__main__":
    print("Fetching commits...")
    all_commits = fetch_commits(max_count=6)

    print("Building or Loading vector index...")
    index, metadata = build_vector_db(all_commits)

    run_streamlit_app(index, metadata)

