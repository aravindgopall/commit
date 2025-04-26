import os
import git
import faiss
import json
import time
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

# --- JIRA Settings ---
ENABLE_JIRA_FETCH = True
JIRA_BASE_URL = os.getenv("JIRA_BASE_URL")
JIRA_AUTH = (os.getenv("JIRA_USER_EMAIL"), os.getenv("JIRA_API_TOKEN"))

# --- Setup Repo ---
REPO_PATH = "path"
repo = git.Repo(REPO_PATH)

NULL_TREE = '4b825dc642cb6eb9a060e54bf8d69288fbee4904'

CACHE_FILE = "commit_embedding_cache.pkl"
ALLOWED_FILE_EXTENSIONS = [".hs"]

# --- Fetch Commits ---
def fetch_commits(max_count=500) -> List[Dict]:
    commits = []
    for commit in repo.iter_commits('main', max_count=max_count):
        commit_hash = commit.hexsha
        changed_files = list(commit.stats.files.keys())

        if not any(file.endswith(tuple(ALLOWED_FILE_EXTENSIONS)) for file in changed_files):
            continue

        try:
            diff_output = subprocess.check_output(
                ["git", "diff", "-W", f"{commit_hash}^", commit_hash],
                cwd=REPO_PATH
            ).decode("utf-8", errors="ignore")
        except subprocess.CalledProcessError:
            diff_output = ""

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

# --- Smart Chunk and Summarize ---
def is_type_declaration(line: str) -> bool:
    keywords = ['data ', 'newtype ', 'type ', 'class ', 'instance ']
    return any(line.lstrip().startswith(k) for k in keywords)

def smart_chunk_diff(diff_text: str, file_extension: str) -> List[str]:
    lines = diff_text.splitlines()
    chunks = []
    current_chunk = ""

    if file_extension != ".hs":
        # fallback for non-Haskell files
        for line in lines:
            if len(current_chunk) + len(line) < 3000:
                current_chunk += line + "\n"
            else:
                chunks.append(current_chunk)
                current_chunk = line + "\n"
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    # Special Haskell logic
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()

        if ("::" in stripped or is_type_declaration(stripped) or stripped.startswith("-- |") or ("=" in stripped and not stripped.startswith("--"))):
            current_chunk += line + "\n"
            i += 1
            brace_balance = line.count("{") - line.count("}")
            while i < len(lines) and (brace_balance > 0 or lines[i].strip() != ""):
                next_line = lines[i]
                current_chunk += next_line + "\n"
                brace_balance += next_line.count("{") - next_line.count("}")
                i += 1
            chunks.append(current_chunk)
            current_chunk = ""
        else:
            i += 1

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def smart_chunk_and_summarize(commit: Dict) -> str:
    diff_text = commit['diff']
    file_extension = next((os.path.splitext(f)[1] for f in commit['files'] if f.endswith(tuple(ALLOWED_FILE_EXTENSIONS))), ".hs")
    chunks = smart_chunk_diff(diff_text, file_extension)

    summaries = []
    for chunk in chunks:
        print("generating summary for chunk:", chunk)
        prompt = f"""
        You are analyzing a Haskell code change.

Write a detailed, structured technical summary that includes:
1. **Module and Imports**: Identify the Haskell module name and any import changes if present.
2. **Changed Entities**: List all functions, types, classes, constants, or instances that were added, removed, or modified.
3. **Change Context and Flow**: Describe the surrounding patterns, conditions (e.g., case matching, guards), and any functional or dispatch flows impacted or newly introduced.
4. **Nature of the Change**: Explain exactly what was changed (e.g., added a new handler for a gateway, added new routing logic, updated validation, introduced error handling, etc.).
5. **Potential System Impact**: Describe if this change could affect other modules, user flows, integrations, or workflows.
6. **Additional Notes**: Highlight if this code introduces new business flows, new service integration points, core assumption changes, or side-effects.

**Goal**: Write the summary to maximize the chances of finding similar changes later through semantic search. Prioritize clarity, completeness, and technical relevance.

Here is the code diff to analyze:
{chunk}
"""
        response = client.chat.completions.create(
            model=DEPLOYMENT_CHAT,
            messages=[{"role": "user", "content": prompt}]
        )
        summaries.append(response.choices[0].message.content.strip())
        print("summary for this: \n", summaries[-1])
    return " ".join(summaries)

# --- Build Vector DB with Per-Commit Cache ---
def build_vector_db(commits: List[Dict]):
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            cache = pickle.load(f)
    else:
        cache = {}

    new_cache = {}
    all_embeddings = []
    all_meta = []

    for commit in commits:
        commit_hash = commit['hash']
        if commit_hash in cache:
            print("using existing vector index for commit:", commit)
            entry = cache[commit_hash]
            embedding = entry['embedding']
            commit_data = entry['meta']
        else:
            print("building vector index for commit:", commit)
            commit_type = classify_commit_llm(commit['message'])
            jira_ids = extract_jira_ids(commit['message'])
            jira_details = " ".join([fetch_jira_description(jid) for jid in jira_ids])
            summary = smart_chunk_and_summarize(commit)

            commit_data = commit.copy()
            commit_data.update({
                'type': commit_type,
                'jira_ids': jira_ids,
                'jira_details': jira_details,
                'summary': summary
            })

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

def embed_text_azure(texts: List[str]) -> np.ndarray:
    response = client.embeddings.create(
        model=DEPLOYMENT_EMBED,
        input=texts
    )
    embeddings = [d.embedding for d in response.data]
    return np.array(embeddings)

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
You are a senior engineer, Always answer based on the context don't answer outside of context if can't answer then simply say can't find relevant commit. Continue this conversation based on previous context and commit summaries.
Previous Conversation:
{context}

New Context from Commits:
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
    st.title("🔎 Commit History Chatbot (Smart Chunking + Memory)")

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
    all_commits = fetch_commits(max_count=2)

    index, metadata = build_vector_db(all_commits)

    run_streamlit_app(index, metadata)

