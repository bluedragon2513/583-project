# Project Repository README

**Authors:** Anthony Minh Nguyen and Noah Sher

The following are explanations of the documents in this project repository.

1. **`project_description_folor`** -- the folder containing files relating to the description of this project.
   
2. **`prompting_document.txt`** —- describes the related files (in the `project_description_folder` directory) and instructions given to the LLM that we used (Claude Sonnet 4.6) to vibe code our project.

3. **`project_report.pdf`** —- a detailed walkthrough of our project implementation.

4. **Error class files** —- the following files describe specific class errors with a detailed analysis of why that is the case so that we could improve our base implementation:
   - `error_class_0_ours.txt` -- manual error class distribution to accommodate LLM's blatant misclassifications.
   - `error_class_1_meta_category_mismatch.txt`
   - `error_class_2_named_entity_overshadow.txt`
   - `error_class_3_wordplay_indirect_category.txt`
   - `error_class_4_right_domain_wrong_entity.txt`

5. **`index_wiki.py`** -- the code that indexes the Wikipedia pages using PyLucene.

6. **`lemmatize.py`** -- the NLTK lemmatization code used at both index and query time.

7. **`search_jeopardy_core.py`** -- the code for our core implementation.

8. **`search_jeopardy_improved.py`** -- the code for our improved implementation.

9. **`run.sh`** -- the script that runs `index_wiki.py` and `search_jeopardy.py` (i.e. the whole project).
