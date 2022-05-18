import os
import json
import stanza
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
nlp = stanza.Pipeline(lang='en', processors='tokenize, pos, lemma, depparse', tokenize_pretokenized=False, tokenize_no_ssplit=True)
tokens_all = set()
root = '/home/Bio/zhangshiqi/codes/two-working/datasets/WebNLG/'

def search(pattern, sequence, direction = 'front'):
    n = len(pattern)
    if direction == 'front':
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
    else:
        res = []
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                res.append(i)
        if res != []:
            return res[-1]
    return -1

def get_deprel(doc):
    deps = []
    for dep in doc.sentences[0].dependencies:
        deps.append((dep[0].id, dep[1], dep[2].id))
    dep_list = [x[0] for x in deps]
    return dep_list, deps

def get_ent_rel(sent, subj, obj, rel_type, subj_type = None, obj_type = None):
    ents = []
    rel = []
    flag = True

    if subj_type == None:
        subj_type = 'Ent'
    if obj_type == None:
        obj_type = 'Ent'

    subj_tokens = [x.text for x in nlp(subj).sentences[0].tokens]
    obj_tokens = [x.text for x in nlp(obj).sentences[0].tokens]

    subj_start = search(subj_tokens, sent, 'back')
    obj_start = search(obj_tokens, sent, 'back')
    if subj_start == -1 or obj_start == -1:
        flag = False
        ents.append([subj_start, subj_start + len(subj_tokens), subj_type, subj_tokens])
        ents.append([obj_start, obj_start + len(obj_tokens), obj_type, obj_tokens])
        rel = [subj_start, subj_start + len(subj_tokens), obj_start, obj_start + len(obj_tokens), rel_type]
    else:
        ents.append([subj_start, subj_start + len(subj_tokens), subj_type])
        ents.append([obj_start, obj_start + len(obj_tokens), obj_type])
        rel = [subj_start, subj_start + len(subj_tokens), obj_start, obj_start + len(obj_tokens), rel_type]
    return ents, rel, flag



def process(path):
    res = []
    error_res = []
    print(path)
    with open(path, 'r') as f:
        conts = json.load(f)

    for cont in tqdm(conts):
        entry = {
            'tokens': [],
            'entities': [],
            'relations': [],
            'deprels': []
        }
        doc = nlp(cont['text'])
        entry['tokens'] = [x.text for x in doc.sentences[0].tokens]
        tokens_all.update(tokens_all.union(set(entry['tokens'])))
        _, dep_rels_with_type = get_deprel(doc)
        entry['deprels'] = [[x[0] - 1, x[0], x[2] - 1, x[2], x[1]] for x in dep_rels_with_type if x[0] - 1 >= 0]
        
        for triple in cont['triple_list']:
            subj, rel_, obj = triple
            ents, rel, flag = get_ent_rel(entry['tokens'], subj, obj, rel_)
            for ent in ents:
                if ent not in entry['entities']:
                    entry['entities'].append(ent)
            entry['relations'].append(rel)
        if flag:
            res.append(entry)
        else:
            error_res.append(entry)

    write_path = '/home/Bio/zhangshiqi/codes/two-working/datasets/unified/' + path.split('/')[-1].split('.')[-2] + '.WebNLGback.json'
    with open(write_path, 'w') as f:
        json.dump(res, f)

    write_path = '/home/Bio/zhangshiqi/codes/two-working/datasets/unified/' + path.split('/')[-1].split('.')[-2] + '.WebNLGback_error.json'
    with open(write_path, 'w') as f:
        json.dump(error_res, f)

        pass

if __name__ == '__main__':
    process(root + 'test.json')
    process(root + 'valid.json')
    process(root + 'train.json')
    # with open('/home/Bio/zhangshiqi/codes/two-working/datasets/WebNLG/tokens_all.json', 'w') as f:
    #     json.dump(list(tokens_all), f)