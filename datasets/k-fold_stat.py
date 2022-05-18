import os
import json


test_sets = []
val_sets = []
train_sets = []
for i in range(10):
    path = f'/home/Bio/zhangshiqi/codes/two-working/datasets/unified/test.ADE{i}.json'
    with open(path, 'r') as f:
        conts = json.load(f)
    
    tmp = set(str(x['tokens']) for x in conts)
    test_sets.append(tmp)

    path = f'/home/Bio/zhangshiqi/codes/two-working/datasets/unified/train.ADE{i}.json'
    with open(path, 'r') as f:
        conts = json.load(f)
    
    tmp = set(str(x['tokens']) for x in conts)
    train_sets.append(tmp)

    path = f'/home/Bio/zhangshiqi/codes/two-working/datasets/unified/valid.ADE{i}.json'
    with open(path, 'r') as f:
        conts = json.load(f)
    
    tmp = set(str(x['tokens']) for x in conts)
    val_sets.append(tmp)

train_all = train_sets[0].union(*train_sets)
val_all = val_sets[0].union(*val_sets)
test_all = test_sets[0].union(*test_sets)

two_all = train_all | val_all | test_all
two_test = test_all


train_sets = []
test_sets = []
for i in range(10):
    path = f'/home/Bio/zhangshiqi/codes/two-working/datasets/ADE/deprel_all/test.ADE{i}.deprel_all.json'
    with open(path, 'r') as f:
        conts = json.load(f)
    tmp = set(str(x['tokens']) for x in conts)
    test_sets.append(tmp)

    path = f'/home/Bio/zhangshiqi/codes/two-working/datasets/ADE/deprel_all/train.ADE{i}.deprel_all.json'
    with open(path, 'r') as f:
        conts = json.load(f)
    tmp = set(str(x['tokens']) for x in conts)
    train_sets.append(tmp)

train_all = train_sets[0].union(*train_sets)
test_all = test_sets[0].union(*test_sets)
pfn_all = train_all | test_all
pfn_test = test_all
pass

