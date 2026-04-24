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
REF_RE = re.compile(r'\[ref\].*?\[/ref\]', re.DOTALL)


def clean_content(text):
    """Strip wiki markup noise, keep readable text."""
    text = TEMPLATE_RE.sub(' ', text)
    text = REF_RE.sub(' ', text)
    text = CATEGORIES_RE.sub(' ', text)
    text = SECTION_RE.sub(' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def parse_wiki_file(filepath):
    """Yield (title, content) pairs from a wiki dump file."""
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
            yield title, content


def build_index():
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])

    index_path = Paths.get(INDEX_DIR)
    directory = FSDirectory.open(index_path)
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
        for title, content in parse_wiki_file(filepath):
            doc = Document()
            # StringField: stored as-is (no analysis) — used to retrieve the title
            doc.add(StringField("title", title, Field.Store.YES))
            # TextField: analyzed (stemmed, stop words removed) — used for search
            doc.add(TextField("content", content, Field.Store.NO))
            # Also index the title text so title words boost retrieval
            doc.add(TextField("title_text", title, Field.Store.NO))
            writer.addDocument(doc)
            total_docs += 1

    writer.commit()
    writer.close()
    print(f"\nIndexed {total_docs:,} documents into '{INDEX_DIR}'")


if __name__ == '__main__':
    build_index()
