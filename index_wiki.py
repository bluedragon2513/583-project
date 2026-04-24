"""
index_wiki.py - Index Wikipedia pages using PyLucene.

Each Wikipedia page (starting with [[Title]]) becomes a separate Lucene document
with two fields: 'title' and 'content'.
"""

import os
import sys
import re
import lucene

from java.nio.file import Paths
from org.apache.lucene.store import FSDirectory
from org.apache.lucene.analysis.en import EnglishAnalyzer
from org.apache.lucene.index import IndexWriter, IndexWriterConfig
from org.apache.lucene.document import Document, Field, TextField, StringField

WIKI_DIR = "wiki-subset-20140602"
INDEX_DIR = "wiki_index"

# Regex to split on page boundaries: [[PageTitle]]
PAGE_BOUNDARY = re.compile(r'^\[\[(.+?)\]\]\s*$', re.MULTILINE)

# Patterns to strip from wiki content before indexing
TEMPLATE_RE = re.compile(r'\[tpl\].*?\[/tpl\]', re.DOTALL)
SECTION_RE = re.compile(r'^==+.*?==+\s*$', re.MULTILINE)
REDIRECT_RE = re.compile(r'^#REDIRECT\s+', re.IGNORECASE)
CATEGORIES_RE = re.compile(r'^CATEGORIES:.*$', re.MULTILINE)
CATEGORIES_EXTRACT_RE = re.compile(r'^CATEGORIES:\s*(.+)$', re.MULTILINE)
REF_RE = re.compile(r'\[ref\].*?\[/ref\]', re.DOTALL)


def clean_content(text):
    """Strip wiki markup noise, keep readable text."""
    text = TEMPLATE_RE.sub(' ', text)
    text = REF_RE.sub(' ', text)
    text = CATEGORIES_RE.sub(' ', text)
    text = SECTION_RE.sub(' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_lead(text):
    """Return the first non-empty paragraph of raw wiki body text.

    Strips markup from templates/refs first, then splits on blank lines and
    returns the first paragraph that has at least 30 characters. Wikipedia
    lead paragraphs define the subject in the same terms Jeopardy clues use.
    """
    text = TEMPLATE_RE.sub(' ', text)
    text = REF_RE.sub(' ', text)
    text = CATEGORIES_RE.sub(' ', text)
    text = SECTION_RE.sub(' ', text)
    for para in re.split(r'\n\s*\n', text):
        para = re.sub(r'\s+', ' ', para).strip()
        if len(para) >= 30:
            return para
    return ''


def extract_categories(text):
    """Return the raw categories string from a wiki body, or empty string."""
    m = CATEGORIES_EXTRACT_RE.search(text)
    return m.group(1).strip() if m else ''


def parse_wiki_file(filepath):
    """Yield (title, content, lead, categories) tuples from a wiki dump file."""
    with open(filepath, encoding='utf-8', errors='replace') as f:
        raw = f.read()

    # Split on [[Title]] markers
    parts = PAGE_BOUNDARY.split(raw)
    # parts = [pre_text, title1, body1, title2, body2, ...]
    # parts[0] is content before the first [[...]], skip it
    i = 1
    while i < len(parts) - 1:
        title = parts[i].strip()
        body = parts[i + 1]
        i += 2
        content = clean_content(body)
        if title and content:
            lead = extract_lead(body)
            cats = extract_categories(body)
            yield title, content, lead, cats


def build_index():
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])

    index_path = Paths.get(INDEX_DIR)
    directory = FSDirectory.open(index_path)
    # EnglishAnalyzer applies a fixed pipeline to every TextField token stream:
    #
    # 1. StandardTokenizer
    #    Splits raw text into word tokens using Unicode text segmentation rules.
    #    Strips punctuation that is not part of a word (periods, commas, etc.)
    #    but keeps internal punctuation like apostrophes and hyphens contextually.
    #    Example: "it's among the top 10" → ["it's", "among", "the", "top", "10"]
    #
    # 2. EnglishPossessiveFilter
    #    Removes trailing "'s" (possessives) from tokens.
    #    Example: "nation's" → "nation",  "Kersee's" → "Kersee"
    #
    # 3. LowerCaseFilter
    #    Lowercases every token so matching is case-insensitive.
    #    Example: "Washington" → "washington",  "UCLA" → "ucla"
    #
    # 4. StopFilter
    #    Removes common English stop words using Lucene's built-in list
    #    (a, an, the, is, in, of, on, at, to, for, with, by, this, that, …).
    #    These words appear in almost every document and carry no discriminating
    #    signal, so removing them reduces index size and improves ranking.
    #    Example: "among the top" → ["top"]  ("among" and "the" are stop words)
    #
    # 5. PorterStemFilter
    #    Applies the Porter stemming algorithm, which strips common English
    #    morphological suffixes to map inflected/derived forms to a shared root.
    #    This is NOT lemmatization: it uses pattern-based rules rather than a
    #    dictionary, so the stem is not always a real word.
    #    Example: "newspapers" → "newspap",  "running" → "run",
    #             "dominant"   → "domin",    "circulation" → "circul"
    #
    # The same EnglishAnalyzer is applied at query time, so query tokens go
    # through the identical pipeline before matching against the index.
    analyzer = EnglishAnalyzer()
    config = IndexWriterConfig(analyzer)
    config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
    writer = IndexWriter(directory, config)

    wiki_files = sorted(
        os.path.join(WIKI_DIR, f)
        for f in os.listdir(WIKI_DIR)
        if f.endswith('.txt')
    )

    total_docs = 0
    for filepath in wiki_files:
        print(f"Indexing {os.path.basename(filepath)}...", flush=True)
        for title, content, lead, cats in parse_wiki_file(filepath):
            doc = Document()
            # StringField: stored as-is (no analysis) — used to retrieve the title
            doc.add(StringField("title", title, Field.Store.YES))
            # TextField: analyzed (stemmed, stop words removed) — used for search
            doc.add(TextField("content", content, Field.Store.NO))
            # Also index the title text so title words boost retrieval
            doc.add(TextField("title_text", title, Field.Store.NO))
            # First paragraph: defining sentence, most relevant to Jeopardy clues
            if lead:
                doc.add(TextField("lead", lead, Field.Store.NO))
            # Wikipedia article categories — matched against Jeopardy category
            if cats:
                doc.add(TextField("categories", cats, Field.Store.NO))
            writer.addDocument(doc)
            total_docs += 1


    writer.commit()
    writer.close()
    print(f"\nIndexed {total_docs:,} documents into '{INDEX_DIR}'")


if __name__ == '__main__':
    build_index()
