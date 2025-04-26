import os
import git
import faiss
import json
import re
import pickle
import subprocess
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
ALLOWED_FILE_EXTENSIONS = [".hs"]  # Configurable list of extensions

def fetch_commits(max_count=500) -> List[Dict]:
    commits = []
    for commit in repo.iter_commits('main', max_count=max_count):
        commit_hash = commit.hexsha
        changed_files = list(commit.stats.files.keys())

        # Filter based on allowed extensions
        if not any(file.endswith(tuple(ALLOWED_FILE_EXTENSIONS)) for file in changed_files):
            continue

        try:
            diff_output = subprocess.check_output(
                ["git", "diff", "-W", f"{commit_hash}^", commit_hash],
                cwd=REPO_PATH
            ).decode("utf-8", errors="ignore")
        except subprocess.CalledProcessError:
            diff_output = ""

        # Further filter diff to only allowed extensions
        filtered_blocks = []
        for block in diff_output.split("diff --git "):
            lines = block.splitlines()
            if lines:
                header_line = lines[0]
                parts = header_line.split()
                if len(parts) >= 2:
                    filename = parts[1].replace("b/", "")
                    if filename.endswith(tuple(ALLOWED_FILE_EXTENSIONS)):
                        filtered_blocks.append(block)

        filtered_diff = "\n".join(filtered_blocks)

        commits.append({
            "hash": commit_hash,
            "author": commit.author.name,
            "date": commit.committed_datetime.isoformat(),
            "message": commit.message.strip(),
            "diff": filtered_diff,
            "files": changed_files,
            "is_merge": len(commit.parents) > 1
        })
    return commits

# --- Helpers ---
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
        if len(current_chunk) + len(line) < 3000:
            current_chunk += line + "\n"
        else:
            chunks.append(current_chunk)
            current_chunk = line + "\n"
    if current_chunk:
        chunks.append(current_chunk)

    summaries = []
    for chunk in chunks:
        print("getting summary for chunk", chunk)
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

# --- Build Vector DB with Per-Commit Cache ---
def build_vector_db(commits: List[Dict]):
    if os.path.exists(CACHE_FILE):
        print("using cache")
        with open(CACHE_FILE, "rb") as f:
            cache = pickle.load(f)
    else:
        cache = {}

    new_cache = {}
    all_embeddings = []
    all_meta = []

    for commit in commits:
        print("building for commit:",commit['hash'])
        commit_hash = commit['hash']
        if commit_hash in cache:
            print("reading from cache")
            entry = cache[commit_hash]
            embedding = entry['embedding']
            commit_data = entry['meta']
        else:
            commit_type = classify_commit_llm(commit['message'])
            jira_ids = extract_jira_ids(commit['message'])
            jira_details = " ".join([fetch_jira_description(jid) for jid in jira_ids])
            print("chunking is started")
            summary = chunk_and_summarize(commit)
            print("chunking is done")

            commit_data = commit.copy()
            commit_data.update({
                'type': commit_type,
                'jira_ids': jira_ids,
                'jira_details': jira_details,
                'summary': summary
            })
            print("embedding start")
            embedding = embed_text_azure([summary + " " + jira_details])[0]

        new_cache[commit_hash] = {
            "meta": commit_data,
            "embedding": embedding
        }

        all_meta.append(commit_data)
        all_embeddings.append(embedding)

    embeddings_np = np.vstack(all_embeddings)
    index = faiss.IndexFlatL2(1536)
    index.add(embeddings_np)

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(new_cache, f)

    return index, all_meta

# --- Search and Answer ---
def search_commits(index, meta, query, top_k=5):
    query_embed = embed_text_azure([query])
    distances, indices = index.search(np.array(query_embed), top_k)
    return [meta[idx] for idx in indices[0] if idx < len(meta)]

def answer_query(index, meta, conversation_history):
    context = "\n".join([
        f"Q: {turn['question']}\nA: {turn['answer']}" for turn in conversation_history[:-1]
    ])

    latest_question = conversation_history[-1]['question']

    commits_context = "\n".join([
        f"Commit Hash: {c['hash']}\nAuthor: {c['author']}\nDate: {c['date']}\nIs Merge: {c['is_merge']}\nMessage: {c['message']}\nJIRA IDs: {','.join(c.get('jira_ids', []))}\nJIRA Details: {c.get('jira_details', '')}\nSummary: {c['summary']}\n"
        for c in search_commits(index, meta, latest_question)
    ])

    prompt = f"""
You are a senior engineer. You are continuing a conversation.
Previous Conversation:
{context}

Now, based on the following commit summaries and metadata, answer the new question.
Context:
{commits_context}

New Question: {latest_question}
Answer:
"""

    response = client.chat.completions.create(
        model=DEPLOYMENT_CHAT,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()

# --- Streamlit Chatbot UI ---
def run_streamlit_app(index, metadata):
    st.title("🔎 Commit History Chatbot (with Memory)")

    if 'conversation' not in st.session_state:
        st.session_state.conversation = []

    if st.button("Clear Conversation"):
        st.session_state.conversation = []

    query = st.text_input("Ask a question about your codebase:")

    if query:
        st.session_state.conversation.append({"question": query, "answer": ""})

        with st.spinner("Thinking..."):
            answer = answer_query(index, metadata, st.session_state.conversation)
            st.session_state.conversation[-1]['answer'] = answer

    for turn in st.session_state.conversation:
        st.markdown(f"**You:** {turn['question']}")
        st.markdown(f"**Bot:** {turn['answer']}")

# --- MAIN FLOW ---
if __name__ == "__main__":
    print("Fetching commits...")
    all_commits = fetch_commits(max_count=6)

    print("Building or Updating vector index...")
    index, metadata = build_vector_db(all_commits)

    print("starting the app...")
    run_streamlit_app(index, metadata)
