import json

types = ['train', 'test', 'valid']
for type in types:
    with open(f'/home/Bio/zhangshiqi/codes/two-working/datasets/unified/{type}.WebNLGfront_error.json', 'r') as f:
        conts = json.load(f)
    with open(f'/home/Bio/zhangshiqi/codes/two-working/datasets/unified/{type}.WebNLGfront.json', 'r') as f:
        conts += json.load(f)

    with open(f'/home/Bio/zhangshiqi/codes/two-working/datasets/WebNLG/done/{type}.WebNLGfront.json', 'w') as f:
        json.dump(conts, f)
        