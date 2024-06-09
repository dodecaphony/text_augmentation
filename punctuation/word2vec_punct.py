import pandas as pd
import gensim
import logging
import urllib.request
import warnings
from gensim.test.utils import datapath
from razdel import tokenize
import pymorphy2
from tqdm import tqdm
import string
import random
import re
import os
import zipfile

tqdm.pandas()
warnings.filterwarnings('ignore')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class W2VPunct:
    """Take path to csv file, substitute lexemes and creates all possible comma placements

       Attributes:
           path: Path to csv file which contains columns sentence_id, raw, correct, type, correct_indexes.
           n_iter: Number of attempts to generate a new sentence.
       """

    def __init__(self, path, n_iter):

        """Download word2vec model for Russian and initialize all necessary variables"""

        urllib.request.urlretrieve("http://vectors.nlpl.eu/repository/20/220.zip",
                                   "ruwikiruscorpora_upos_cbow_300_10_2021.bin.gz")
        self.df = pd.read_csv(path, sep=';').head(20)
        self.curr_dir = os.path.abspath(os.path.dirname(__file__))
        path_to_zip_file = os.path.join(self.curr_dir, 'ruwikiruscorpora_upos_cbow_300_10_2021.bin.gz')
        directory_to_extract_to = os.path.join(self.curr_dir, 'ruwikiruscorpora_upos_cbow_300_10_2021.bin')
        with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
            zip_ref.extractall(directory_to_extract_to)
        self.path_model = os.path.join(self.curr_dir, 'ruwikiruscorpora_upos_cbow_300_10_2021.bin/model.bin')
        self.model = gensim.models.KeyedVectors.load_word2vec_format(datapath(self.path_model), binary=True)
        self.morph = pymorphy2.MorphAnalyzer()
        self.sentences = []
        self.generated_pairs = []
        self.n_iter = n_iter
        self.final_pairs = []

    def preprocess(self, text):

        """Cleans numbers, applies morphological parser and unifies POS tags """

        parsed_sent = {'text': text, 'tokens': {}}
        res = text.replace('(1)', 'NUM')
        res = res.replace('(2)', 'NUM')
        res = res.replace('(3)', 'NUM')
        res = res.replace('(4)', 'NUM')
        res = res.replace('(5)', 'NUM')
        res = res.replace('(6)', 'NUM')
        res = res.replace('(7)', 'NUM')
        res = res.replace('(8)', 'NUM')
        res = res.replace('(9)', 'NUM')
        tokens = list(tokenize(res))
        for i in range(len(tokens)):
            p = self.morph.parse(tokens[i].text)[0]
            if not p.normal_form is None:
                if p.tag.POS == 'ADJF' or p.tag.POS == 'ADJS':
                    s = str(p.normal_form) + '_' + 'ADJ'
                elif p.tag.POS == 'ADVB':
                    s = str(p.normal_form) + '_' + 'ADV'
                elif p.tag.POS == 'PREP':
                    s = str(p.normal_form) + '_' + 'ADP'
                elif p.tag.POS == 'INFN':
                    s = str(p.normal_form) + '_' + 'VERB'
                elif p.tag.POS == 'NPRO':
                    s = str(p.normal_form) + '_' + 'PRON'
                else:
                    s = str(p.normal_form) + '_' + str(p.tag.POS)

            else:
                s = ''
            parsed_sent['tokens'].update({i: {'text': tokens[i].text, 'text_pos': s, 'pos': p.tag.POS, 'feats': p.tag}})
        self.sentences.append(parsed_sent)

    def generate_sentence(self, sentence, to_replace):

        """Replaces lexemes with similar lexemes and inflects new lexeme"""

        inflected = {}
        for ID, value in sentence['tokens'].items():
            for idx, new_word in to_replace.items():
                if ID == idx:
                    s2 = self.morph.parse(new_word)[0]
                    for s1 in self.morph.parse(value['text']):
                        gramemmes = set(s1.tag.grammemes)
                        if s1.tag.POS in s2.tag:
                            s3 = s2.inflect(gramemmes)
                            if s3 is not None:
                                inflected.update({ID: s3.word})
                        break
        if not inflected:
            print('Cannot generate new sentence due to problems with inflection errors')
            return None
        else:
            result_tokens = [None] * len(sentence['tokens'].keys())
            for ID, value in sentence['tokens'].items():
                for idx, new_word in inflected.items():
                    if ID == idx:
                        result_tokens[ID] = new_word
                    else:
                        result_tokens[ID] = value['text']
            res = "".join([" " + i if i != '«' and i != '»' and i not in string.punctuation else i for i in
                           result_tokens]).strip()
            return " ".join(res.strip().split())

    def replace_words_in_sent(self, sentence):

        """Checks if it possible to replace any lexeme"""

        id_possible_tokens = []
        for ID, value in sentence['tokens'].items():
            if value['text_pos'] in self.model:
                id_possible_tokens.append(ID)
        to_replace = {}
        for idx in id_possible_tokens:
            for ID, value in sentence['tokens'].items():
                if ID == idx:
                    model_answ = list(self.model.most_similar(value['text_pos']))
                    model_answ_pos = []
                    for word in model_answ:
                        pos_w = word[0].split('_')[1]
                        pos_orig = value['text_pos'].split('_')[1]
                        if pos_w == pos_orig:
                            model_answ_pos.append(word[0].split('_')[0])
                    try:
                        word_to_replace = random.choice(model_answ_pos)
                        to_replace.update({ID: word_to_replace})
                    except:
                        pass
        return self.generate_sentence(sentence, to_replace)

    def replace_num(self, corr):

        """Replacing NUM with numbers"""

        try:
            corr_split = corr.split()
            i = 1
            for j in range(len(corr_split)):
                if corr_split[j] == 'NUM':
                    corr_split[j] = '(' + str(i) + ')'
                    i += 1
            return ' '.join(corr_split)
        except:
            pass

    def generate_variants(self, id_sent, text1, text2, task, correct_indexes):

        """Generates all possible comma placements in the sentence"""

        regex = r'\(\d\)'
        matches = re.findall(regex, correct_indexes, re.MULTILINE)
        res = text2
        if len(list(matches)) != 0:
            for ind in matches:
                res = res.replace(ind, ' , ')
        res = res.replace('(1)', ' ')
        res = res.replace('(2)', ' ')
        res = res.replace('(3)', ' ')
        res = res.replace('(4)', ' ')
        res = res.replace('(5)', ' ')
        res = res.replace('(6)', ' ')
        res = res.replace('(7)', ' ')
        res = res.replace('(8)', ' ')
        res = res.replace('(9)', ' ')
        res = " ".join(res.strip().split())
        res = res.replace(' ,', ',')
        res = res.replace(' .', '.')
        res = res.replace(' !', '!')
        res = res.replace(' ?', '?')
        self.final_pairs.append([id_sent, text1, res, task, correct_indexes])

    def corrupt(self):

        """Main function that aggregates all processes"""

        print('Preprocessing...')
        self.df['raw'].progress_apply(lambda x: self.preprocess(x))
        print('Replacing words...')
        for sent in tqdm(self.sentences):
            for i in range(self.n_iter):
                corrupt_sent = self.replace_words_in_sent(sent)
                if [sent, corrupt_sent] not in self.generated_pairs:
                    self.generated_pairs.append([sent['text'], corrupt_sent])
        df_gen = pd.DataFrame(self.generated_pairs, columns=['original', 'corrected'])
        df_gen = df_gen.drop_duplicates()
        df_gen_with_answers = df_gen.merge(self.df, how='left', left_on='original', right_on='raw')
        df_gen_with_answers = df_gen_with_answers[['sentence_id', 'original', 'corrected', 'type', 'correct_indexes']]
        print('Generating all possible comma placements...')
        df_gen_with_answers['generated_with_nums'] = df_gen_with_answers['corrected'].progress_apply(
            lambda x: self.replace_num(x))

        df_gen_with_answers.progress_apply(
            lambda x: self.generate_variants(x['sentence_id'], x['original'], x['generated_with_nums'], x['type'],
                                        x['correct_indexes']), axis=1)
        corrected_sent_from_generated = pd.DataFrame(self.final_pairs,
                                                     columns=['sentence_id', 'original_with_commas', 'generated',
                                                              'task', 'correct_indexes'])
        return corrected_sent_from_generated
