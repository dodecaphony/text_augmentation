`word2vec_punct.py`

An algorithm that takes as input pairs of sentences from the collected data, and replaces lexemes with semantically closest ones using the word2vec model

Folder `punctuation`:

- `raw_pairs_with_numbers.csv` - parsed raw data from mentioned source
- `generated_pairs.tsv` - all possible comma placements from raw pairs
- `raw_pairs_with_mumbers.csv` - the result of applying `word2vec_punct.py` to raw data (raw_pairs_with_numbers)
