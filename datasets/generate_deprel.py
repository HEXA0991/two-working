import json
from mimetypes import types_map
import stanza
import os
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
nlp = stanza.Pipeline(lang='en', processors='tokenize, pos, lemma, depparse', tokenize_pretokenized=True)

# dataset = 'CoNLL04'
# dataset = 'ADE0'
# train = f'/home/Bio/zhangshiqi/codes/two-working/datasets/unified/train.{dataset}.json'
# dev = f'/home/Bio/zhangshiqi/codes/two-working/datasets/unified/valid.{dataset}.json'
# test = f'/home/Bio/zhangshiqi/codes/two-working/datasets/unified/test.{dataset}.json'
# prune = 1

def search(pattern, sequence):
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1

def _generate_dep_label():
    conts = 'acl - acl:relcl - advcl - advmod - amod - appos - aux - aux:pass - case - cc - cc:preconj - ccomp - compound - compound:prt - conj - cop - csubj - csubj:pass - dep - det - det:predet - discourse - dislocated - expl - fixed - flat - flat:foreign - goeswith - iobj - list - mark - nmod - nmod:npmod - nmod:poss - nmod:tmod - nsubj - nsubj:pass - nummod - obj - obl - obl:npmod - obl:tmod - orphan - parataxis - punct - reparandum - root - vocative - xcomp'
    res = conts.split(' - ')
    res = {x:0 for x in res}
    return res

def head_to_lca(head, prune, subj_pos, obj_pos):
    """
    Convert a sequence of head indexes into a tree object.
    """
    root = None
    len_ = len(head)

    if prune < 0:
        nodes = [Tree() for _ in head]  # prune should NOT < 0

        for i in range(len(nodes)):
            h = head[i]
            nodes[i].idx = i
            nodes[i].dist = -1 # just a filler
            if h == 0:
                root = nodes[i]
            else:
                nodes[h-1].add_child(nodes[i])
    else:
        # find dependency path
        # subj_pos = [i for i in range(len_) if subj_pos[i] == 0]
        # obj_pos = [i for i in range(len_) if obj_pos[i] == 0]

        cas = None

        subj_ancestors = set(subj_pos)
        # 找subj的祖先以及祖先的祖先一直到根节点
        for s in subj_pos:
        # for s in [x - 1 for x in subj_pos]:
            h = head[s]
            tmp = [s]
            while h > 0:
                tmp += [h-1]    # stanford id -1 -> head id
                subj_ancestors.add(h-1)
                h = head[h-1]

            if cas is None:
                cas = set(tmp)  # cas中是-1过的list中的index
            else:
                cas.intersection_update(tmp)

        obj_ancestors = set(obj_pos)
        for o in obj_pos:
        # for o in [x - 1 for x in obj_pos]:
            h = head[o]
            tmp = [o]
            while h > 0:
                tmp += [h-1]
                obj_ancestors.add(h-1)
                h = head[h-1]
            cas.intersection_update(tmp)

        # find lowest common ancestor
        if len(cas) == 1:
            lca = list(cas)[0]
        else:
            child_count = {k:0 for k in cas}
            for ca in cas:
                if head[ca] > 0 and head[ca] - 1 in cas:    # ca不是根 and ca的head也是ca
                    child_count[head[ca] - 1] += 1  # 这个ca的head（也是ca）多一个孩子

            # the LCA has no child in the CA set
            for ca in cas:
                if child_count[ca] == 0:
                    lca = ca    # 如果这个ca没有孩子那就是lca
                    break
        #提出假设，subj和obj祖先到某个地方就全部一致了 
        path_nodes = subj_ancestors.union(obj_ancestors)
        # 到这已经找到lca了但是还得计算是几跳的lca，符不符合标准
        # compute distance to path_nodes
        candidate = []  # 不能保证prune > 1 情况下也好用
        for s in subj_pos:
            cnt = 0
            while cnt < prune:
                cnt += 1
                if head[s] - 1 not in candidate and head[s] - 1:
                    candidate.append(head[s] - 1)
        for o in obj_pos:
            cnt = 0
            while cnt < prune:
                cnt += 1
                if head[o] - 1 in candidate:
                    candidate.append(head[o] - 1)
        res = [x for x in path_nodes if x in candidate and x not in subj_pos and x not in obj_pos]
        return res

def get_deprel(tokens):
    doc = nlp([tokens])
    deps = []
    for dep in doc.sentences[0].dependencies:
        deps.append((dep[0].id, dep[1], dep[2].id))
    dep_list = [x[0] for x in deps]
    return dep_list, deps

def read_json(path):
    with open(path, 'r') as f:
        conts = json.load(f)
    return conts

def get_lcas(path):
    conts = read_json(path)
    print(path)
    for cont in tqdm(conts):
        deps = get_deprel(cont['tokens'])
        dep_rels, dep_rels_with_type = []
        for rel in cont['relations']:
            s_start, s_end, o_start, o_end, rel_type = rel
            s_span = list(range(s_start, s_end))
            o_span = list(range(o_start, o_end))
            lcas = head_to_lca(deps, prune, s_span, o_span)
            for lca in lcas:
                if [lca,lca + 1,'Mid'] not in cont['entities']:
                    cont['entities'].append([lca,lca + 1,'Mid'])
                if [lca, lca + 1, s_start, s_end, 'Dep_To'] not in dep_rels:
                    dep_rels.append([lca, lca + 1, s_start, s_end, 'Dep_To'])
                if [lca, lca + 1, o_start, o_end, 'Dep_To'] not in dep_rels:
                    dep_rels.append([lca, lca + 1, o_start, o_end, 'Dep_To'])
                    
        cont['relations'] += dep_rels
    new_path = path.strip('.json') + '.deprel+rel.json'
    # new_path = path.strip('.json') + '.deprel.json'
    with open(new_path, 'w') as f:
        json.dump(conts, f)

def write_deps_all(path):
    conts = read_json(path)
    for cont in tqdm(conts):
        _, dep_rels_with_type = get_deprel(cont['tokens'])
        deprels = [[x[0] - 1, x[0], x[2] - 1, x[2], x[1]] for x in dep_rels_with_type if x[0] - 1 >= 0]
        cont['deprels'] = deprels
    new_path = '/home/Bio/zhangshiqi/codes/two-working/datasets/SCIERC/deprel_all/' + path.split('/')[-1].strip('.json') + '.deprel_all.json'
    with open(new_path, 'w') as f:
        json.dump(conts, f)



if __name__ == '__main__':
    # get_lcas(train)
    # get_lcas(dev)
    # get_lcas(test)
    # dataset = 'ADE'
    # for i in range(10):
    #     train = f'/home/Bio/zhangshiqi/codes/two-working/datasets/unified/train.{dataset}{i}.json'
    #     dev = f'/home/Bio/zhangshiqi/codes/two-working/datasets/unified/valid.{dataset}{i}.json'
    #     test = f'/home/Bio/zhangshiqi/codes/two-working/datasets/unified/test.{dataset}{i}.json'
    #     print(dataset + str(i))
    #     write_deps_all(train)
    #     write_deps_all(dev)
    #     write_deps_all(test)
    dataset = 'SCIERC'
    train = f'/home/Bio/zhangshiqi/codes/two-working/datasets/SCIERC/twoed/train.{dataset}.json'
    dev = f'/home/Bio/zhangshiqi/codes/two-working/datasets/SCIERC/twoed/valid.{dataset}.json'
    test = f'/home/Bio/zhangshiqi/codes/two-working/datasets/SCIERC/twoed/test.{dataset}.json'
    write_deps_all(train)
    write_deps_all(dev)
    write_deps_all(test)
    pass