# Project Repository README

**Authors:** Anthony Minh Nguyen and Noah Sher

The following are explanations of the documents in this project repository.

1. **`prompting_document.txt`** —- describes the related files (in the `project_description_folder` directory) and instructions given to the LLM that we used (Claude Sonnet 4.6) to vibe code our project.

2. **`project_experiments.pdf`** -- all of our notes/results before, during, and after each iteration of our implementations.

3. **`project_report.pdf`** —- a detailed walkthrough of our project implementation.

4. **Error class files** —- the following files describe specific class errors with a detailed analysis of why that is the case so that we could improve our base implementation:
   - `error_class_1_meta_category_mismatch.txt`
   - `error_class_2_product_vs_parent.txt`
   - `error_class_3_quote_attribution.txt`
   - `error_class_4_wordplay_indirect.txt`
   - `error_class_5_wrong_specific_entity.txt`

5. **`index_wiki.py`** -- the code that indexes the Wikipedia pages using PyLucene.

6. **`lemmatize.py`** -- the NLTK lemmatization code used at both index and query time.

7. **`search_jeopardy.py`** -- the code that retrieves Wikipedia page titles for Jeopardy clues.

8. **`run.sh`** -- the script that runs `index_wiki.py` and `search_jeopardy.py` (i.e. the whole project).
