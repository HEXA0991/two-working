import os
import json

def process(path):
    with open(path, 'r') as f:
        conts = json.load(f)
    res = []
    for cont in conts:
        entry = {
            'tokens': cont['tokens'],
            'entities': [],
            'relations': []
        }

        for ent in cont['entities']:
            entry['entities'].append([ent['start'], ent['end'], ent['type']])
        for rel in cont['relations']:
            entry['relations'].append([
                entry['entities'][rel['head']][0], 
                entry['entities'][rel['head']][1], 
                entry['entities'][rel['tail']][0], 
                entry['entities'][rel['tail']][1], 
                rel['type']
            ])
        res.append(entry)
    return res

if __name__ == '__main__':
    cor_types = ['train', 'test', 'dev']
    for cor_type in cor_types:
        path = f'/home/Bio/zhangshiqi/codes/two-working/datasets/SCIERC/{cor_type}_triples.json'
        write_path = f'/home/Bio/zhangshiqi/codes/two-working/datasets/SCIERC/twoed/{cor_type}.SCIERC.json'
        conts = process(path)
        with open(write_path, 'w') as f:
            json.dump(conts, f)
