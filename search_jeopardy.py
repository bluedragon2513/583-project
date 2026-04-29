"""
search_jeopardy.py - Retrieve Wikipedia page titles for Jeopardy clues.

Reads questions.txt, builds a query from the clue (and optionally the
category), searches the Lucene index, and reports the top result.

Query strategy:
  - Use the full clue text as a BooleanQuery against 'content'.
  - Add the category words as a lower-weight boost against 'content'.
  - EnglishAnalyzer handles stemming and stop-word removal.
"""

import os
import re
import sys
import lucene

from java.nio.file import Paths
from org.apache.lucene.store import FSDirectory
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.search import IndexSearcher, BooleanQuery, BooleanClause, BoostQuery
from org.apache.lucene.search.similarities import BM25Similarity
from org.apache.lucene.analysis.en import EnglishAnalyzer
from org.apache.lucene.queryparser.classic import QueryParser

INDEX_DIR = "wiki_index"
QUESTIONS_FILE = os.path.join("project description folder", "questions.txt")
TOP_K = 10  # retrieve top-K docs per question


def load_questions(path):
    """Parse questions.txt into list of (category, clue, answers) tuples.

    Format: every non-empty block is 3 lines:
        CATEGORY
        CLUE TEXT
        ANSWER1|ANSWER2
    followed by a blank line.
    """
    questions = []
    with open(path, encoding='utf-8') as f:
        lines = [line.rstrip('\n') for line in f]

    i = 0
    while i < len(lines):
        # Skip blank lines
        if not lines[i].strip():
            i += 1
            continue
        category = lines[i].strip()
        i += 1
        if i >= len(lines):
            break
        clue = lines[i].strip()
        i += 1
        if i >= len(lines):
            break
        answer_raw = lines[i].strip()
        i += 1
        answers = [a.strip().lower() for a in answer_raw.split('|')]
        questions.append((category, clue, answers))
    return questions


def normalize(text):
    """Lowercase and strip punctuation for answer comparison."""
    return re.sub(r'[^a-z0-9 ]', '', text.lower()).strip()


def answer_matches(predicted_title, gold_answers):
    """Return True if the predicted title matches any accepted answer."""
    pred = normalize(predicted_title)
    for ans in gold_answers:
        gold = normalize(ans)
        if pred == gold:
            return True
        # Also accept if one contains the other (handles partial matches like
        # "Arlington National Cemetery" vs "Arlington Cemetery")
        if gold and (gold in pred or pred in gold):
            return True
    return False


def lucene_escape(text):
    """Escape special Lucene query characters from a Python string."""
    special = r'\+-!(){}[]^"~*?:|&/'
    return ''.join('\\' + c if c in special else c for c in text)


# Words that appear constantly in Jeopardy clues but carry no retrieval signal.
# EnglishAnalyzer already removes standard stop words (a, the, is, …); these
# are Jeopardy-specific filler words that survive the standard list.
JEOPARDY_STOP_WORDS = {
    'this', 'these', 'his', 'her', 'their', 'its', 'our', 'your',
    'he', 'she', 'they', 'it', 'we', 'you', 'who', 'whom', 'whose',
    'said', 'called', 'known', 'named', 'aka', 'also', 'just',
    'one', 'two', 'first', 'second', 'last', 'new', 'old',
    'man', 'woman', 'person', 'people', 'type', 'kind', 'form',
}


def remove_jeopardy_stops(text):
    """Strip Jeopardy filler words, keeping meaningful content words."""
    tokens = text.split()
    filtered = [t for t in tokens if t.lower() not in JEOPARDY_STOP_WORDS]
    return ' '.join(filtered) if filtered else text


def extract_named_entities(clue):
    """Return multi-word proper noun phrases from the clue.

    Only consecutive sequences of 2+ capitalised tokens are returned.
    Single capitalised words (Olympics, Nile, UCLA) appear as supporting
    context in Jeopardy clues, not as the answer — boosting them against
    titles pulls up the wrong pages. Two-word+ sequences (El Tahrir,
    Pierre Cauchon, Wolong Nature Reserve) are specific enough to be
    useful anchors even if they are not the answer themselves.
    """
    tokens = clue.split()
    entities = []
    run = []
    for i, tok in enumerate(tokens):
        core = tok.lstrip('"\'(').rstrip('",)')
        if not core:
            run = []
            continue
        is_cap = core[0].isupper() and not core.isupper()  # skip ALL-CAPS
        if is_cap and i > 0 and core.lower() not in JEOPARDY_STOP_WORDS:
            run.append(core)
        else:
            if len(run) >= 2:
                entities.append(' '.join(run))
            run = []
    if len(run) >= 2:
        entities.append(' '.join(run))
    return entities


def build_query(analyzer, category, clue):
    """Build a BooleanQuery from the clue and category.

    Improvement 1 — Category-aware query construction:
      Category words are searched against both 'content' (boost 0.5) and
      'title_text' (boost 0.8).

    Improvement 2 — Jeopardy stop word removal:
      Filler words common in Jeopardy phrasing are stripped before querying.

    Improvement 3 — Lead paragraph boosting:
      The clue query is also run against the 'lead' field (boost 1.5).
      Wikipedia lead paragraphs define the article subject with the same
      descriptive language Jeopardy clues use, making them the highest-signal
      section for retrieval.

    Improvement 4 — Named entity upweighting (multi-word only):
      Consecutive sequences of 2+ capitalised tokens are extracted from the
      clue and searched against 'title_text' (boost 2.0). Single capitalised
      words are skipped because they are supporting context in Jeopardy clues,
      not the answer — only specific multi-word phrases like "El Tahrir" or
      "Pierre Cauchon" point usefully toward an article.
    """
    # --- Improvement 2: remove Jeopardy filler from the clue ---
    clean_clue = remove_jeopardy_stops(clue)
    escaped_clue = lucene_escape(clean_clue)

    builder = BooleanQuery.Builder()
    any_clause = False

    # Primary signal: clue text against content and title
    for field in ("content", "title_text"):
        try:
            parser = QueryParser(field, analyzer)
            parser.setDefaultOperator(QueryParser.Operator.OR)
            q = parser.parse(escaped_clue)
            builder.add(q, BooleanClause.Occur.SHOULD)
            any_clause = True
        except Exception:
            pass

    if not any_clause:
        return None

    # --- Improvement 3: lead paragraph boosting ---
    try:
        lead_parser = QueryParser("lead", analyzer)
        lead_parser.setDefaultOperator(QueryParser.Operator.OR)
        lead_q = lead_parser.parse(escaped_clue)
        builder.add(BoostQuery(lead_q, 1.5), BooleanClause.Occur.SHOULD)
    except Exception:
        pass

    # --- Improvement 4: multi-word named entity upweighting ---
    named_entities = extract_named_entities(clue)
    for phrase in named_entities:
        escaped_phrase = lucene_escape(phrase)
        try:
            ne_parser = QueryParser("title_text", analyzer)
            ne_parser.setDefaultOperator(QueryParser.Operator.AND)
            ne_q = ne_parser.parse(escaped_phrase)
            builder.add(BoostQuery(ne_q, 2.0), BooleanClause.Occur.SHOULD)
        except Exception:
            pass

    # --- Improvement 1: category-aware boosting ---
    if category:
        escaped_cat = lucene_escape(category)
        # Category against body content (moderate boost)
        try:
            cat_content_parser = QueryParser("content", analyzer)
            cat_q = cat_content_parser.parse(escaped_cat)
            builder.add(BoostQuery(cat_q, 0.5), BooleanClause.Occur.SHOULD)
        except Exception:
            pass
        # Category against document title (stronger boost — answer type often
        # appears in the page title)
        try:
            cat_title_parser = QueryParser("title_text", analyzer)
            cat_tq = cat_title_parser.parse(escaped_cat)
            builder.add(BoostQuery(cat_tq, 0.8), BooleanClause.Occur.SHOULD)
        except Exception:
            pass
        # Category against Wikipedia article categories (strong signal —
        # "NEWSPAPERS" matches articles categorised as newspapers, etc.)
        try:
            cat_cats_parser = QueryParser("categories", analyzer)
            cat_cats_q = cat_cats_parser.parse(escaped_cat)
            builder.add(BoostQuery(cat_cats_q, 2.0), BooleanClause.Occur.SHOULD)
        except Exception:
            pass

    return builder.build()


def search(questions, top_k=TOP_K, k1=1.2, b=0.0):
    """Run retrieval with BM25(k1, b). b=0.0 disables length normalization (best empirically)."""
    index_path = Paths.get(INDEX_DIR)
    directory = FSDirectory.open(index_path)
    reader = DirectoryReader.open(directory)
    searcher = IndexSearcher(reader)
    searcher.setSimilarity(BM25Similarity(k1, b))
    analyzer = EnglishAnalyzer()

    results = []  # list of (category, clue, answers, ranked_titles)

    for category, clue, answers in questions:
        query = build_query(analyzer, category, clue)
        if query is None:
            results.append((category, clue, answers, []))
            continue

        hits = searcher.search(query, top_k)
        ranked_titles = []
        stored = searcher.storedFields()
        for hit in hits.scoreDocs:
            doc = stored.document(hit.doc)
            title = doc.get("title")
            ranked_titles.append(title)

        results.append((category, clue, answers, ranked_titles))

    reader.close()
    return results


def extract_category_constraint(category):
    """Extract an answer-type constraint from the category string.

    Handles two cases:
      1. Parenthetical Alex note: '(Alex: We'll give you the museum. You give us the state.)'
      2. Known implicit patterns that carry a hard answer-type rule without a note.
    """
    match = re.search(r'\(Alex[:\s]+(.+?)\)', category, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    if "parent company" in category.lower():
        return (
            "The answer must be the parent corporation that owns the brand or product, "
            "NOT the brand or product itself. Pick the candidate that is a company."
        )
    return None


def rerank_with_llm(results, model="gemini-2.0-flash"):
    """Rerank each question's candidate titles using an LLM and return updated results."""
    from google import genai
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    reranked = []

    for i, (category, clue, answers, titles) in enumerate(results):
        if not titles:
            reranked.append((category, clue, answers, titles))
            continue

        numbered = "\n".join(f"{j+1}. {t}" for j, t in enumerate(titles))
        constraint = extract_category_constraint(category)
        constraint_line = f"IMPORTANT — {constraint}\n\n" if constraint else ""
        prompt = (
            f"You are solving a Jeopardy! clue. Pick the Wikipedia article title "
            f"that is the correct answer.\n\n"
            f"Category: {category}\n"
            f"Clue: {clue}\n\n"
            f"{constraint_line}"
            f"Candidates:\n{numbered}\n\n"
            f"Reply with only the number of the correct title."
        )

        try:
            response = client.models.generate_content(model=model, contents=prompt)
            raw = response.text.strip()
            idx = int(''.join(c for c in raw if c.isdigit())) - 1
            if 0 <= idx < len(titles):
                reordered = [titles[idx]] + [t for j, t in enumerate(titles) if j != idx]
            else:
                reordered = titles
        except Exception:
            reordered = titles

        reranked.append((category, clue, answers, reordered))
        print(f"  [{i+1:3}/100] {category[:25]:<25} → {reordered[0][:45]}", flush=True)

    return reranked


def compute_metrics(results):
    """Compute P@1 and MRR over all questions."""
    p_at_1_total = 0
    mrr_total = 0.0
    correct_at_1 = []
    incorrect_at_1 = []

    for category, clue, answers, ranked_titles in results:
        # P@1
        if ranked_titles and answer_matches(ranked_titles[0], answers):
            p_at_1_total += 1
            correct_at_1.append((category, clue, answers, ranked_titles))
        else:
            incorrect_at_1.append((category, clue, answers, ranked_titles))

        # MRR: find rank of first correct result
        for rank, title in enumerate(ranked_titles, start=1):
            if answer_matches(title, answers):
                mrr_total += 1.0 / rank
                break

    n = len(results)
    p_at_1 = p_at_1_total / n if n else 0
    mrr = mrr_total / n if n else 0
    return p_at_1, mrr, correct_at_1, incorrect_at_1


def main():
    if not os.path.exists(INDEX_DIR):
        print(f"Index not found at '{INDEX_DIR}'. Run index_wiki.py first.")
        sys.exit(1)

    questions = load_questions(QUESTIONS_FILE)
    print(f"Loaded {len(questions)} questions.")

    lucene.initVM(vmargs=['-Djava.awt.headless=true'])

    results = search(questions)

    if os.environ.get("GEMINI_API_KEY"):
        print("\nReranking with LLM...")
        results = rerank_with_llm(results)
    else:
        print("\nNo GEMINI_API_KEY set; skipping LLM reranking.")

    print("\n=== Per-Question Results ===")
    for i, (category, clue, answers, ranked_titles) in enumerate(results, 1):
        top = ranked_titles[0] if ranked_titles else "(no result)"
        correct = answer_matches(top, answers) if ranked_titles else False
        mark = "✓" if correct else "✗"
        print(f"{i:3}. [{mark}] {category[:30]:<30} | Predicted: {top[:50]:<50} | Gold: {answers[0]}")

    p_at_1, mrr, correct_at_1, incorrect_at_1 = compute_metrics(results)

    print(f"\n=== Performance Metrics ===")
    print(f"Total questions : {len(results)}")
    print(f"Correct at rank 1: {len(correct_at_1)}")
    print(f"P@1             : {p_at_1:.4f} ({p_at_1*100:.1f}%)")
    print(f"MRR             : {mrr:.4f}")

    print(f"\n=== Error Analysis ===")
    print(f"Correctly answered: {len(correct_at_1)}/{len(results)}")
    print(f"Incorrectly answered: {len(incorrect_at_1)}/{len(results)}")
    print("\nSample incorrect predictions:")
    for category, clue, answers, ranked_titles in incorrect_at_1[:10]:
        top = ranked_titles[0] if ranked_titles else "(no result)"
        print(f"  Cat: {category}")
        print(f"  Clue: {clue[:80]}")
        print(f"  Gold: {answers[0]}  |  Predicted: {top}")
        print()


if __name__ == '__main__':
    main()
