import json

re = []
ner = []

with open('/home/Bio/zhangshiqi/codes/two-working/datasets/WebNLG/deprel_all/train.WebNLGfront.deprel_all.json', 'r') as f:
    conts = json.load(f)

for cont in conts:
    for rel in cont['relations']:
        if rel[-1] not in re:
            re.append(rel[-1])

print(len(re))