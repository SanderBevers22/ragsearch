import os
from dotenv import load_dotenv
from google import genai

def spell_check(query):
    prompt=f"""Fix any spelling errors in the user-provided movie search query below.
    Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
    Preserve punctuation and capitalization unless a change is required for a typo fix.
    If there are no spelling errors, or if you're unsure, output the original query unchanged.
    Output only the final query text, nothing else.
    User query: "{query}"
    """   
    return prompt

def rewrite_query(query):
    prompt = f"""Rewrite the user-provided movie search query below to be more specific and searchable.

    Consider:
    - Common movie knowledge (famous actors, popular films)
    - Genre conventions (horror = scary, animation = cartoon)
    - Keep the rewritten query concise (under 10 words)
    - It should be a Google-style search query, specific enough to yield relevant results
    - Don't use boolean logic

    Examples:
    - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
    - "movie about bear in london with marmalade" -> "Paddington London marmalade"
    - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

    If you cannot improve the query, output the original unchanged.
    Output only the rewritten query text, nothing else.

    User query: "{query}"
    """
    return prompt

def expand_search(query):
    prompt = f"""Expand the user-provided movie search query below with related terms.

    Add synonyms and related concepts that might appear in movie descriptions.
    Keep expansions relevant and focused.
    Output the original query and the additional terms as this is an expanded search.

    Examples:
    - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
    - "action movie with bear" -> "action thriller bear chase fight adventure"
    - "comedy with bear" -> "comedy funny bear humor lighthearted"

    User query: "{query}"
    """
    return prompt

def individual_reranking(query,doc):
    prompt = f"""Rate how well this movie matches the search query.

    Query: "{query}"
    Movie: {doc.get("title", "")} - {doc.get("description", "")}

    Consider:
    - Direct relevance to query
    - User intent (what they're looking for)
    - Content appropriateness

    Rate 0-10 (10 = perfect match).
    Give me ONLY the number in your response, no other text or explanation.

    Score:"""

    return prompt

def batch_reranking(query,docs):
    doc_list = "\n".join(f"{doc['id']}: {doc['title']} - {doc['description'][:120]}" for doc in docs)

    prompt = f"""Rank the movies listed below by relevance to the following search query.

    Query: "{query}"

    Movies:
    {doc_list}

    Return ONLY the movie IDs in order of relevance (best match first). Return a valid JSON list, nothing else.

    For example:
    [75, 12, 34, 2, 1]

    Ranking:"""

    return prompt

def enhance_query(prompt):
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")

    response = client.models.generate_content(model="gemma-3-27b-it",contents=prompt)

    usage = response.usage_metadata
    print(f"\n Prompt tokens: {usage.prompt_token_count}")
    print(f"Response tokens: {usage.candidates_token_count}")

    return response.text.strip()
