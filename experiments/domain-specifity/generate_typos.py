import random

from analyze_typos import bea_data, ruspellru_data, jfleg_data, bea_errors, ruspellru_errors, jfleg_errors


neighbors_ru = {
    'й': ['ц', 'ф'],
    'ц': ['й', 'у', 'ф', 'ы'],
    'у': ['ц', 'к', 'ы', 'в'],
    'к': ['у', 'е', 'в', 'а'],
    'е': ['к', 'н', 'а', 'п'],
    'н': ['е', 'г', 'п', 'р'],
    'г': ['н', 'ш', 'р', 'о'],
    'ш': ['г', 'щ', 'о', 'л'],
    'щ': ['ш', 'з', 'л', 'д'],
    'з': ['щ', 'х', 'д', 'ж'],
    'х': ['з', 'ъ', 'ж', 'э'],
    'ф': ['й', 'ц', 'ы', 'в', 'а'],
    'ы': ['ц', 'у', 'ф', 'в', 'а', 'п'],
    'в': ['у', 'к', 'ф', 'ы', 'а', 'п'],
    'а': ['к', 'е', 'ф', 'ы', 'в', 'п'],
    'п': ['е', 'н', 'ф', 'ы', 'в', 'а'],
    'р': ['н', 'г', 'о', 'л'],
    'о': ['г', 'ш', 'р', 'л'],
    'л': ['ш', 'щ', 'г', 'о', 'д'],
    'д': ['щ', 'з', 'л', 'г', 'ж'],
    'ж': ['з', 'х', 'д', 'л', 'э'],
    'э': ['х', 'ж', 'ъ'],
    'я': ['ч', 'с'],
    'ч': ['я', 'м', 'с', 'и'],
    'с': ['я', 'ч', 'м', 'и', 'ть'],
    'м': ['ч', 'с', 'и', 'ь'],
    'и': ['ч', 'с', 'м', 'ть', 'б'],
    'ь': ['т', 'о', 'л', 'б'],
    'б': ['и', 'ь', 'ю'],
    'ю': ['д', 'б', 'ж'],
    'ъ': ['х', 'э'],
}


neighbors_en = {
    'q': ['w', 'a'],
    'w': ['q', 'e', 'a', 's'],
    'e': ['w', 'r', 's', 'd'],
    'r': ['e', 't', 'd', 'f'],
    't': ['r', 'y', 'f', 'g'],
    'y': ['t', 'u', 'g', 'h'],
    'u': ['y', 'i', 'h', 'j'],
    'i': ['u', 'o', 'j', 'k'],
    'o': ['i', 'p', 'k', 'l'],
    'p': ['o', 'l'],
    'a': ['q', 'w', 's', 'z'],
    's': ['w', 'e', 'a', 'd', 'z', 'x'],
    'd': ['e', 'r', 's', 'f', 'x', 'c'],
    'f': ['r', 't', 'd', 'g', 'c', 'v'],
    'g': ['t', 'y', 'f', 'h', 'v', 'b'],
    'h': ['y', 'u', 'g', 'j', 'b', 'n'],
    'j': ['u', 'i', 'h', 'k', 'n', 'm'],
    'k': ['i', 'o', 'j', 'l', 'm'],
    'l': ['o', 'p', 'k'],
    'z': ['a', 's', 'x'],
    'x': ['s', 'd', 'z', 'c'],
    'c': ['d', 'f', 'x', 'v'],
    'v': ['f', 'g', 'c', 'b'],
    'b': ['g', 'h', 'v', 'n'],
    'n': ['h', 'j', 'b', 'm'],
    'm': ['j', 'k', 'n'],
}


def introduce_typo(sentence, error_type, language='en'):
    if len(sentence) == 0:
        return sentence

    # Выбор словаря с соседями в зависимости от языка
    if language == 'en':
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        neighbors = neighbors_en
    elif language == 'ru':
        alphabet = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
        neighbors = neighbors_ru
    else:
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        neighbors = {}

    index = random.randint(0, len(sentence) - 1)
    char = sentence[index]

    if error_type == 'TInsertion':
        # Вставка случайного символа
        return sentence[:index] + random.choice(alphabet) + sentence[index:]
    elif error_type == 'TDeletion':
        # Удаление символа
        return sentence[:index] + sentence[index + 1:]
    elif error_type == 'TSubstitution':
        # Замена символа на соседний, если он есть в словаре соседей
        if char in neighbors:
            return sentence[:index] + random.choice(neighbors[char]) + sentence[index + 1:]
        else:
            return sentence[:index] + random.choice(alphabet) + sentence[index + 1:]
    elif error_type == 'TTransposition' and len(sentence) > 1:
        # Транспозиция символов
        if index == len(sentence) - 1:
            index -= 1
        return sentence[:index] + sentence[index + 1] + sentence[index] + sentence[index + 2:]
    elif error_type == 'TReplication':
        # Дублирование символа
        return sentence[:index] + sentence[index] + sentence[index:]
    else:
        return sentence


def generate_synthetic_data(data, error_distribution, num_samples_per_sentence=1, language='en'):
    synthetic_sentences = []
    total_errors = sum(error_distribution.values())
    error_probs = [count / total_errors for count in error_distribution.values()]

    for _, row in data.iterrows():
        sentence = row['incorrect']
        for _ in range(num_samples_per_sentence):
            error_type = random.choices(list(error_distribution.keys()), weights=error_probs, k=1)[0]
            synthetic_sentence = introduce_typo(sentence, error_type, language)
            synthetic_sentences.append(synthetic_sentence)
    return synthetic_sentences


def save_to_file(synthetic_data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for sentence in synthetic_data:
            file.write(sentence + '\n')

# Старый подход
# error_distribution = {
#     'TInsertion': 0.2,
#     'TDeletion': 0.2,
#     'TSubstitution': 0.2,
#     'TTransposition': 0.2,
#     'TReplication': 0.2
# }


# Генерация синтетических данных с использованием полученного распределения
ruspellru_synthetic = generate_synthetic_data(ruspellru_data, ruspellru_errors, language='ru')
jfleg_synthetic = generate_synthetic_data(jfleg_data, jfleg_errors, language='en')
bea_synthetic = generate_synthetic_data(bea_data, bea_errors, language='en')

ruspellru_synthetic_path = 'synthetic_datasets/ruspellru_synthetic.txt'
jfleg_synthetic_path = 'synthetic_datasets/jfleg_synthetic.txt'
bea_synthetic_path = 'synthetic_datasets/bea_synthetic.txt'

save_to_file(ruspellru_synthetic, ruspellru_synthetic_path)
save_to_file(jfleg_synthetic, jfleg_synthetic_path)
save_to_file(bea_synthetic, bea_synthetic_path)
