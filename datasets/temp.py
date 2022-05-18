import json

def get(path):
    with open(path, 'r') as f:
        conts = json.load(f)
    return set(str(x['tokens']) for x in conts)

if __name__ == '__main__':
    front = get('/home/Bio/zhangshiqi/codes/two-working/datasets/unified/test.WebNLGfront_error.json')
    front_back = get('/home/Bio/zhangshiqi/codes/two-working/datasets/unified/test.WebNLGfront_back_error.json')
    back = get('/home/Bio/zhangshiqi/codes/two-working/datasets/unified/test.WebNLGback_error.json')
    pass