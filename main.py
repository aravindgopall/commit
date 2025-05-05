import os
import git
import subprocess
import pickle
import re
import json
import numpy as np
import requests
from collections import defaultdict
import streamlit as st
from openai import AzureOpenAI
from typing import List, Dict, Any, Optional, Union
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.models import PointStruct, Distance, VectorParams


client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)
DEPLOYMENT_CHAT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
DEPLOYMENT_EMBED = os.getenv("AZURE_OPENAI_DEPLOYMENT_EMBED")


QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "git_hunks")
EMBEDDING_DIM = 1536  


ENABLE_JIRA_FETCH = True
JIRA_BASE_URL = os.getenv("JIRA_BASE_URL")
JIRA_AUTH = (os.getenv("JIRA_USER_EMAIL"), os.getenv("JIRA_API_TOKEN"))


REPO_PATH = "/Users/pramod.p/euler-api-gateway/"
repo = git.Repo(REPO_PATH)
ALLOWED_FILE_EXTENSIONS = [".hs", ".py", ".ts", ".go"]


HIERARCHY_CACHE_FILE = "intent_hierarchy_cache.pkl"


def init_qdrant_client():
    """Initialize Qdrant client and ensure collection exists"""
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    

    collections = qdrant_client.get_collections().collections
    collection_names = [c.name for c in collections]
    
    if QDRANT_COLLECTION not in collection_names:
        qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        print(f"Created new Qdrant collection: {QDRANT_COLLECTION}")
    
    return qdrant_client


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


def fetch_commits(max_count):
    commits = []
    i = 1
    for commit in repo.iter_commits('staging', max_count=max_count):
        print(f"fetching commit {i}/{max_count}")
        commit_hash = commit.hexsha
        changed_files = list(commit.stats.files.keys())
        if not any(file.endswith(tuple(ALLOWED_FILE_EXTENSIONS)) for file in changed_files):
            continue

        try:
            diff_output = subprocess.check_output(
                ["git", "diff", "-U0", f"{commit_hash}^", commit_hash],
                cwd=REPO_PATH
            ).decode("utf-8", errors="ignore")
        except subprocess.CalledProcessError:
            diff_output = ""

        jira_ids = extract_jira_ids(commit.message)
        jira_details = " ".join([fetch_jira_description(jid) for jid in jira_ids])
        commit_type = classify_commit_llm(commit.message)

        commits.append({
            "hash": commit_hash,
            "message": commit.message.strip(),
            "diff": diff_output,
            "author": commit.author.name,
            "date": commit.committed_datetime.isoformat(),
            "jira_ids": jira_ids,
            "jira_details": jira_details,
            "type": commit_type
        })
        i+=1
    return commits

def fetch_given_commits(commit_hashes: List[str]) -> List[Dict]:
    """Fetch specific commits (provided manually) and their diffs from the repository"""
    commits = []
    for i, commit_hash in enumerate(commit_hashes, start=1):
        print(f"fetching commit {i}/{len(commit_hashes)}: {commit_hash}")
        
        try:
            commit = repo.commit(commit_hash)
        except Exception as e:
            print(f"Error fetching commit {commit_hash}: {e}")
            continue
        
        changed_files = list(commit.stats.files.keys())
        if not any(file.endswith(tuple(ALLOWED_FILE_EXTENSIONS)) for file in changed_files):
            print("skipped one")
            continue
        
        try:
            diff_output = subprocess.check_output(
                ["git", "diff", "-W", f"{commit_hash}^", commit_hash],
                cwd=REPO_PATH
            ).decode("utf-8", errors="ignore")
        except subprocess.CalledProcessError:
            diff_output = ""

        jira_ids = extract_jira_ids(commit.message)
        jira_details = " ".join([fetch_jira_description(jid) for jid in jira_ids])
        commit_type = classify_commit_llm(commit.message)

        commits.append({
            "hash": commit_hash,
            "message": commit.message.strip(),
            "diff": diff_output,
            "author": commit.author.name,
            "date": commit.committed_datetime.isoformat(),
            "jira_ids": jira_ids,
            "jira_details": jira_details,
            "type": commit_type
        })
    return commits



def split_diff_into_hunks(diff_text: str) -> List[str]:
    hunks = []
    current_hunk = []
    for line in diff_text.splitlines():
        if line.startswith('@@'):
            if current_hunk:
                hunks.append('\n'.join(current_hunk))
                current_hunk = []

        if not line.startswith('-'):
            current_hunk.append(line)
    if current_hunk:
        hunks.append('\n'.join(current_hunk))
    return hunks


def detect_hunk_intent(hunk_text: str) -> Dict[str, str]:
    """
    Enhanced function to detect the intent of a hunk with detailed explanation.
    Returns both a short intent summary and a detailed explanation of the workflow.
    """
    prompt = f"""
You are analyzing a code diff hunk.

Please provide:
1. A SHORT INTENT summary of this hunk in 1 short sentence (less than 10 words).
2. A DETAILED EXPLANATION of the workflow and purpose of these changes (3-5 sentences).

Focus on both high-level intent and technical details.

Here is the hunk:
{hunk_text}

Format your response as:
SHORT_INTENT: [your short intent here]
DETAILED_EXPLANATION: [your detailed explanation here]
"""
    response = client.chat.completions.create(
        model=DEPLOYMENT_CHAT,
        messages=[{"role": "user", "content": prompt}]
    )
    
    result_text = response.choices[0].message.content.strip()
    

    short_intent = ""
    detailed_explanation = ""
    
    lines = result_text.split('\n')
    for line in lines:
        if line.startswith("SHORT_INTENT:"):
            short_intent = line[len("SHORT_INTENT:"):].strip().lower()
        elif line.startswith("DETAILED_EXPLANATION:"):
            detailed_explanation = line[len("DETAILED_EXPLANATION:"):].strip()
    

    if not short_intent:
        short_intent = result_text.lower()
        detailed_explanation = result_text
    
    return {
        "short_intent": short_intent,
        "detailed_explanation": detailed_explanation
    }


def group_hunks_by_intent(commits: List[Dict]) -> Dict[str, List[Dict]]:
    grouped = defaultdict(list)
    for i, commit in enumerate(commits):
        print(f"Processing commit {i+1}/{len(commits)}")
        hunks = split_diff_into_hunks(commit['diff'])
        for idx, hunk in enumerate(hunks):
            if not hunk.strip():
                continue
            
            intent_data = detect_hunk_intent(hunk)
            short_intent = intent_data["short_intent"]
            detailed_explanation = intent_data["detailed_explanation"]
            
            grouped[short_intent].append({
                "commit": commit,
                "hunk": hunk,
                "hunk_id": f"{commit['hash']}_{idx}",
                "intent": short_intent,
                "detailed_explanation": detailed_explanation,
                "jira_ids": commit.get('jira_ids', []),
                "jira_details": commit.get('jira_details', ''),
                "commit_type": commit.get('type', 'unknown')
            })
    return grouped


def parse_alternate_response(response: str):
    """
    Parses the structured response from LLM into a dictionary of categories and their intents.
    
    Example response:
    Category: Transaction Handling
    Intents:
    - add handling for verify.nb parameter types in transaction processing
    - add pre-transaction validation logic

    Returns a dictionary of categories with intents.
    """
    categories = {}
    current_category = None
    current_intents = []


    lines = response.split('\n')

    for line in lines:
        line = line.strip()  # Clean up whitespace

        if line.startswith("Category:"):

            if current_category:
                categories[current_category] = current_intents


            current_category = line[len("Category:"):].strip()
            current_intents = []

        elif line.startswith("-"):

            current_intents.append(line[len("-"):].strip())


    if current_category:
        categories[current_category] = current_intents

    return categories


def group_intents_with_llm(grouped_hunks: Dict[str, List[Dict]], predefined_categories=None):
    """
    Create or update higher-level categories from existing lower-level intents using LLM.
    Made more generic by allowing optional predefined categories and customizable grouping logic.
    
    Args:
        grouped_hunks: Dictionary of intents mapped to their hunks
        predefined_categories: Optional list of predefined category names to guide the grouping
    
    Returns:
        Dictionary mapping high-level categories to their constituent intents
    """
    intent_hierarchy = {}


    all_intents = list(grouped_hunks.keys())


    batch_size = 15
    num_batches = (len(all_intents) + batch_size - 1) // batch_size

    for i in range(0, len(all_intents), batch_size):
        batch_intents = all_intents[i:i + batch_size]
        batch_num = i // batch_size + 1
        print(f"Grouping new intents: batch {batch_num}/{num_batches}")


        intent_list = "\n".join([f"- {intent}" for intent in batch_intents])
        

        if predefined_categories:
            categories_list = "\n".join([f"- {cat}" for cat in predefined_categories])
            prompt = f"""
You are an expert software architect analyzing code change intents.

Below is a list of specific code change intents:
{intent_list}

Group these intents into the following predefined categories:
{categories_list}

For each category, list which of the original intents belong to that category.
Only use the predefined categories - do not create new ones.

Please provide the response in the following format:

Category: [Category Name]
Intents:
- [intent1]
- [intent2]
- [intent3]
...
"""
        else:
            prompt = f"""
You are an expert software architect analyzing code change intents.

Below is a list of specific code change intents:
{intent_list}

Group these intents into 2-4 meaningful, generic categories. For each category:
1. Create a concise, domain-agnostic name (3-5 words).
2. List which of the original intents belong to that category.
3. Ensure categories are appropriately abstract and could apply across different codebases.

Please provide the response in the following format:

Category: [Category Name]
Intents:
- [intent1]
- [intent2]
- [intent3]
...
"""

        try:

            response = client.chat.completions.create(
                model=DEPLOYMENT_CHAT,
                messages=[{"role": "user", "content": prompt}]
            )


            if response.choices and response.choices[0].message.content.strip():
                result = parse_alternate_response(response.choices[0].message.content.strip())


                for category, intents in result.items():
                    if category in intent_hierarchy:
                        intent_hierarchy[category].extend(intents)
                    else:
                        intent_hierarchy[category] = intents
            else:
                print(f"Empty or invalid response for batch {batch_num}. Skipping...")

        except Exception as e:
            print(f"Error during API call for batch {batch_num}: {e}. Skipping this batch.")

    return intent_hierarchy


def embed_texts_azure(texts: List[str]) -> np.ndarray:
    response = client.embeddings.create(
        model=DEPLOYMENT_EMBED,
        input=texts
    )
    embeddings = [d.embedding for d in response.data]
    return np.array(embeddings)


def build_qdrant_db(qdrant_client, grouped_hunks: Dict[str, List[Dict]]):
    """
    Build or update the Qdrant collection with hunk embeddings and metadata.
    Embeds both the short intent and detailed explanation.
    """

    try:
        existing_points = qdrant_client.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=qmodels.Filter(),
            limit=10000  # Adjust if you have more points
        )[0]
        existing_ids = {p.id for p in existing_points}
        print(f"Found {len(existing_ids)} existing points in Qdrant")
    except Exception as e:
        print(f"Error checking existing points: {e}")
        existing_ids = set()


    points_to_upsert = []
    all_metadata = []
    

    for intent, hunks in grouped_hunks.items():
        for hunk_info in hunks:
            hunk_id = hunk_info['hunk_id']

            numerical_id = int(hash(hunk_id) % (10**10))
            

            if numerical_id in existing_ids:
                print(f"Skipping existing hunk_id: {hunk_id}")
                continue
            

            detailed_explanation = hunk_info.get('detailed_explanation', "")
            

            metadata_dict = {
                "intent": intent,
                "hunk_id": hunk_id,
                "commit_hash": hunk_info['commit']['hash'],
                "commit_message": hunk_info['commit']['message'],
                "commit_author": hunk_info['commit']['author'],
                "commit_date": hunk_info['commit']['date'],
                "jira_ids": ",".join(hunk_info.get('jira_ids', [])),
                "jira_details": hunk_info.get('jira_details', ''),
                "commit_type": hunk_info.get('commit_type', 'unknown'),
                "detailed_explanation": detailed_explanation,
                "hunk_text": hunk_info['hunk']
            }
            

            all_metadata.append(metadata_dict)
            

            points_to_upsert.append({
                "id": numerical_id,
                "payload": metadata_dict,
                "intent": intent,
                "detailed_explanation": detailed_explanation,
                "hunk_id": hunk_id
            })
    

    if points_to_upsert:

        batch_size = 100  # Adjust based on API limits
        for i in range(0, len(points_to_upsert), batch_size):
            batch = points_to_upsert[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(points_to_upsert) + batch_size - 1) // batch_size}")
            

            composite_texts = []
            for p in batch:

                if p["detailed_explanation"]:
                    composite_text = f"Intent: {p['intent']}\nDetailed explanation: {p['detailed_explanation']}"
                else:
                    composite_text = p["intent"]
                composite_texts.append(composite_text)
            

            embeddings = embed_texts_azure(composite_texts)
            

            points = [
                PointStruct(
                    id=p["id"],
                    vector=embeddings[j].tolist(),
                    payload=p["payload"]
                )
                for j, p in enumerate(batch)
            ]
            

            qdrant_client.upsert(
                collection_name=QDRANT_COLLECTION,
                points=points
            )
            print(f"Upserted {len(points)} points to Qdrant")
    
    return all_metadata


def search_with_llm_hierarchy(qdrant_client, metadata, hierarchy, query, top_k=5):
    """
    Search using the LLM-generated hierarchy, first checking if query exactly matches a category,
    then falling back to Qdrant vector search if not.
    """

    categories_list = list(hierarchy.keys())
    

    exact_category_match = None
    for category in categories_list:

        if query.lower() == category.lower():
            exact_category_match = category
            break
    
    if not exact_category_match:

        categories_str = ", ".join(categories_list)
        
        prompt = f"""
You have a list of code change categories:
{categories_str}

Does the query "{query}" EXACTLY match any of these categories? 
Only respond with the exact category name if it's a precise match.
If it's related but not an exact match, respond with "None".
If it's not related at all, respond with "None".

Response:
"""
        response = client.chat.completions.create(
            model=DEPLOYMENT_CHAT,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50
        )
        
        category_match = response.choices[0].message.content.strip()
        

        if category_match in hierarchy:
            exact_category_match = category_match
    
    if exact_category_match:

        print(f"Found exact category match: {exact_category_match}")
        matching_intents = hierarchy[exact_category_match]
        

        result_points = []
        for intent in matching_intents:

            intent_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="intent",
                        match=qmodels.MatchText(text=intent)
                    )
                ]
            )
            
            response = qdrant_client.scroll(
                collection_name=QDRANT_COLLECTION,
                scroll_filter=intent_filter,
                limit=top_k // len(matching_intents) + 1  # Distribute top_k across intents
            )[0]
            
            result_points.extend([p.payload for p in response])
        

        if len(result_points) > top_k:

            composite_texts = []
            for item in result_points:

                intent = item["intent"]
                detailed_explanation = item.get("detailed_explanation", "")
                if detailed_explanation:
                    composite_text = f"Intent: {intent}\nDetailed explanation: {detailed_explanation}"
                else:
                    composite_text = intent
                composite_texts.append(composite_text)
                
            item_embeddings = embed_texts_azure(composite_texts)
            query_embedding = embed_texts_azure([query])[0]
            

            similarities = np.dot(item_embeddings, query_embedding)
            sorted_indices = np.argsort(-similarities)  # Descending order
            

            result_points = [result_points[idx] for idx in sorted_indices[:top_k]]
        
        return result_points
    else:

        print("No exact category match, falling back to embedding search")
        query_embedding = embed_texts_azure([query])[0]
        

        search_result = qdrant_client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=query_embedding.tolist(),
            limit=top_k
        )
        

        return [point.payload for point in search_result]


def generate_developer_response(query: str, hunks_info: List[Dict], max_tokens=7000) -> str:
    combined_hunks = ""
    token_count = 0
    base_prompt_tokens = 500
    query_tokens = len(query) // 4
    available_tokens = max_tokens - base_prompt_tokens - query_tokens
    
    for item in hunks_info:
        item_text = f"Hunk:\n{item['hunk_text']}\n"
        if item.get('commit_type'):
            item_text += f"Commit Type: {item.get('commit_type')}\n"
        if item.get('jira_ids'):
            item_text += f"JIRA IDs: {item.get('jira_ids')}\n"
        if item.get('jira_details'):
            item_text += f"JIRA Details: {item.get('jira_details')}\n"
        if item.get('detailed_explanation'):
            item_text += f"Detailed Explanation: {item.get('detailed_explanation')}\n"
        item_text += f"Intent: {item.get('intent', '')}\n\n"
        
        item_tokens = len(item_text) // 4
        if token_count + item_tokens > available_tokens:
            combined_hunks += "Note: Some hunks were omitted due to token limits.\n"
            break
            
        combined_hunks += item_text
        token_count += item_tokens
    
    prompt = f"""
You are a senior Haskell developer assistant.

You are given:
- A user request describing a task.
- Related code diffs (hunks) with associated JIRA information, commit types, and intent descriptions.

Here is the user task:
{query}

Here are the related code hunks with metadata:
{combined_hunks}

Your job is to:

1. **Explain the code hunks**: Summarize what the provided code hunks are doing in clear, simple English.
2. **Formulate a plan**: Based on the task, explain the steps needed to implement the task using the information from the code hunks. Write the steps in simple English.
3. **Generate code**: Write the Haskell code needed to complete the task.
4. **Specify file locations**: For each code snippet you generate, mention the file path where it should be placed. (Use the file path information from the git diff.)

Answer in the following format:

---

**Explanation of the provided code hunks:**
<your explanation>

**Steps to complete the task:**
<list of steps>

**Generated Code and File Paths:**
- File: `<file_path_1>`
```haskell
<code_snippet_1>
:
"""
    response = client.chat.completions.create(
        model=DEPLOYMENT_CHAT,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


@st.cache_data
def load_commits(max_count):
    return fetch_commits(max_count)

@st.cache_data
def load_grouped_hunks(commits):
    return group_hunks_by_intent(commits)

@st.cache_data
def load_intent_hierarchy(grouped_hunks):
    return group_intents_with_llm(grouped_hunks)

@st.cache_resource
def load_qdrant_client():
    return init_qdrant_client()

@st.cache_data
def load_qdrant_metadata(_qdrant_client, grouped_hunks):
    return build_qdrant_db(_qdrant_client, grouped_hunks)


def run_streamlit_app(qdrant_client, metadata, intent_hierarchy):
    st.title("🔍 Git Diff Hunks Chatbot (Qdrant + Hierarchical Intent)")

    if 'conversation' not in st.session_state:
        st.session_state.conversation = []

    st.sidebar.header("Filters")
    commit_types = ["all", "feature", "bugfix", "refactor", "documentation", "test", "other"]
    selected_type = st.sidebar.selectbox("Filter by commit type", commit_types)


    st.sidebar.header("Intent Categories")
    for high_level, low_levels in intent_hierarchy.items():
        with st.sidebar.expander(f"📁 {high_level}"):
            for intent in low_levels:
                st.write(f"- {intent}")

    if st.sidebar.button("🔄 Refresh All Data"):
        st.cache_data.clear()
        st.cache_resource.clear()

        try:
            qdrant_client.delete_collection(collection_name=QDRANT_COLLECTION)
            print(f"Deleted Qdrant collection: {QDRANT_COLLECTION}")
        except Exception as e:
            print(f"Error deleting collection: {e}")

        if os.path.exists(HIERARCHY_CACHE_FILE):
            os.remove(HIERARCHY_CACHE_FILE)
        st.experimental_rerun()

    query = st.text_input("Ask a question about your codebase:")

    if query:
        st.session_state.conversation.append({"question": query, "answer": ""})

        with st.spinner("Thinking..."):
            relevant_hunks = search_with_llm_hierarchy(qdrant_client, metadata, intent_hierarchy, query)
            if selected_type != "all":
                relevant_hunks = [h for h in relevant_hunks if h.get('commit_type') == selected_type]
            if not relevant_hunks:
                st.session_state.conversation[-1]['answer'] = "No relevant code changes found."
            else:
                answer = generate_developer_response(query, relevant_hunks)
                st.session_state.conversation[-1]['answer'] = answer

    for turn in st.session_state.conversation:
        st.markdown(f"**You:** {turn['question']}")
        st.markdown(f"**Bot:** {turn['answer']}")


def main():
    st.sidebar.title("⚙️ Settings")
    max_commits = st.sidebar.slider("Number of commits to analyze", 10, 200, 50)
    
    with st.spinner("Loading and indexing commits..."):
        commits = load_commits(max_count=3)
        print(f"Commits loaded: {len(commits)}")
    
    with st.spinner("Splitting diffs into hunks and grouping by intent..."):
        grouped_hunks = load_grouped_hunks(commits)
        print(f"Unique intents found: {len(grouped_hunks)}")
    
    with st.spinner("Building LLM-based intent hierarchy..."):
        intent_hierarchy = load_intent_hierarchy(grouped_hunks)
        print(f"Created {len(intent_hierarchy)} high-level categories")
    
    with st.spinner("Building Qdrant vector database..."):
        qdrant_client = load_qdrant_client()

        metadata = load_qdrant_metadata(qdrant_client, grouped_hunks)
        print(f"Vector database built with {len(metadata)} entries")

    run_streamlit_app(qdrant_client, metadata, intent_hierarchy)


if __name__ == "__main__":
    main()