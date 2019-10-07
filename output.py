import re
import random

split_mark_pattern=re.compile(r'[\(、。・＿；：」…「＾）℃（！？,!\?\.\[\]『』【】《》\)＊\n＜＞]')
split_digit=re.compile(r'[\d]+')

# 一行ずつパターンにマッチした記号と数字を処理する関数
def split_marks(x):
    try:
        x=split_mark_pattern.sub('',x)
        x=split_digit.sub('0',x)
        return x
    except:
        return x


def output(mes,source_ids,target_words,model,tokenizer_obj):

    input_data = split_marks(mes)
    
    print("\nPredict")
    query=janome_parse(input_data,tokenizer_obj)

    res=translate_one(query,source_ids,target_words,model)

    try:
        manzai_response = re.sub('0',str(random.randint(1,100)),''.join(res).replace('<unk>',''))
    except:
        manzai_response = ''.join(res).replace('<unk>','')

    return manzai_response

def janome_parse(text,tokenizer_obj):
    tokens=tokenizer_obj.tokenize(text)
    return [token.surface for token in tokens]


def translate_one(source,source_ids,target_words,model):
    words = source
    print('# source : ' + ' '.join(words))
    x = model.xp.array(
        [source_ids.get(w, 1) for w in words], 'i')
    ys = model.translate([x], beam=5)[0]
    words = [target_words[y] for y in ys]
    print('#  result : ' + ' '.join(words))
    return words