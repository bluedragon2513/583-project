"""
lemmatize.py - Shared NLTK lemmatization used at both index and query time.

Applied as a pre-processing step BEFORE Lucene's EnglishAnalyzer so that the
full pipeline is:

  raw text
    -> lemmatize_text()          [this module, Python/NLTK]
         1. word_tokenize        split into word tokens
         2. pos_tag              assign Penn-Treebank POS tags (NN, VB, JJ, RB…)
         3. WordNetLemmatizer    reduce each word to its dictionary base form
                                 using the POS tag to choose the correct paradigm
    -> EnglishAnalyzer           [Lucene, applied automatically at index & search]
         4. StandardTokenizer    re-tokenize on Unicode word boundaries
         5. EnglishPossessiveFilter  strip trailing 's
         6. LowerCaseFilter      lowercase
         7. StopFilter           remove English stop words
         8. PorterStemFilter     light suffix stripping on already-lemmatized forms

Why lemmatize before stemming?
  - Lemmatization handles irregular morphology that Porter stemming cannot:
      went  -> go    (Porter leaves it as "went")
      better -> good  (Porter leaves it as "better")
      ran   -> run   (Porter leaves it as "ran")
  - After lemmatization the forms are already normalised, so Porter stemming
    has little additional work to do and rarely distorts the token.

Why keep EnglishAnalyzer after lemmatizing?
  - Stop word removal still needed (lemmatization doesn't remove "the", "a", …)
  - Lowercasing still needed for consistent matching
  - Light Porter pass handles plural/suffix noise the lemmatizer may miss
    on rare or unknown words

POS mapping (Penn Treebank -> WordNet):
  NN*/NNS  -> NOUN    (default for unknown tags)
  VB*      -> VERB
  JJ*      -> ADJ
  RB*      -> ADV
"""

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet

_lemmatizer = WordNetLemmatizer()


def _penn_to_wordnet(penn_tag):
    """Convert a Penn Treebank POS tag to the WordNet equivalent."""
    if penn_tag.startswith('V'):
        return wordnet.VERB
    if penn_tag.startswith('J'):
        return wordnet.ADJ
    if penn_tag.startswith('R'):
        return wordnet.ADV
    return wordnet.NOUN  # default: noun


def lemmatize_text(text):
    """Return a lemmatized version of text, preserving token order.

    Each word is mapped to its WordNet base form using its part-of-speech tag
    to choose the correct lemma paradigm.

    Examples:
        "The newspapers were running stories"
            -> "The newspaper be run story"
        "She went to better schools"
            -> "She go to good school"
    """
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    lemmas = [
        _lemmatizer.lemmatize(word, _penn_to_wordnet(tag))
        for word, tag in tagged
    ]
    return ' '.join(lemmas)
