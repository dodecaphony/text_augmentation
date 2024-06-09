import pandas as pd
from collections import Counter


ruspellru_incorrect_path = 'typo_datasets/ruspellru_sources.txt'
ruspellru_correct_path = 'typo_datasets/ruspellru_corrections.txt'
jfleg_incorrect_path = 'typo_datasets/jfleg_sources.txt'
jfleg_correct_path = 'typo_datasets/jfleg_corrections.txt'
bea_incorrect_path = 'typo_datasets/bea_sources.txt'
bea_correct_path = 'typo_datasets/bea_corrections.txt'


def load_data(incorrect_path, correct_path):
    with open(incorrect_path, 'r', encoding='utf-8') as file:
        incorrect_sentences = file.readlines()
    with open(correct_path, 'r', encoding='utf-8') as file:
        correct_sentences = file.readlines()

    assert len(incorrect_sentences) == len(correct_sentences), "err, lens not matching"

    data = pd.DataFrame({
        'incorrect': [sentence.strip() for sentence in incorrect_sentences],
        'correct': [sentence.strip() for sentence in correct_sentences]
    })
    return data


def classify_and_count_errors(data):
    error_types = {
        'TDeletion': 0,
        'TInsertion': 0,
        'TReplication': 0,
        'TSubstitution': 0,
        'TTransposition': 0
    }

    for _, row in data.iterrows():
        inc, cor = row['incorrect'], row['correct']

        len_inc, len_cor = len(inc), len(cor)

        if len_inc + 1 == len_cor and cor.startswith(inc):
            error_types['TDeletion'] += 1
            continue

        if len_inc - 1 == len_cor and inc.startswith(cor):
            error_types['TInsertion'] += 1
            continue

        if abs(len_inc - len_cor) > 1:
            continue

        if len_inc == len_cor:
            diff_chars = [(c1, c2) for c1, c2 in zip(inc, cor) if c1 != c2]
            if len(diff_chars) == 1:
                error_types['TSubstitution'] += 1
            elif len(diff_chars) == 2:
                if (diff_chars[0][0] == diff_chars[1][1]) and (diff_chars[0][1] == diff_chars[1][0]):
                    error_types['TTransposition'] += 1
            elif len(diff_chars) > 2:
                continue

            if len_inc > len_cor:
                for i in range(len_inc - 1):
                    if inc[i] == inc[i + 1] and inc[:i] + inc[i + 1:] == cor:
                        error_types['TReplication'] += 1
                        break

    return error_types


ruspellru_data = load_data(ruspellru_incorrect_path, ruspellru_correct_path)
jfleg_data = load_data(jfleg_incorrect_path, jfleg_correct_path)
bea_data = load_data(bea_incorrect_path, bea_correct_path)

ruspellru_errors = classify_and_count_errors(ruspellru_data)
jfleg_errors = classify_and_count_errors(jfleg_data)
bea_errors = classify_and_count_errors(bea_data)

print("RuSpellRu", ruspellru_errors)
print("JFleg", jfleg_errors)
print("Bea", bea_errors)
