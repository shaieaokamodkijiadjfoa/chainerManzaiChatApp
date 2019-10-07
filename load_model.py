import dill
from manzai_model import ParseArg,Transformer
from chainer import cuda,serializers
import chainer
import numpy as np
import os
from janome.tokenizer import Tokenizer

def load_model():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_name = os.path.basename(os.getcwd())
    data_dir= os.path.join(BASE_DIR, f'{ project_name }/data')

    if cuda.available and cuda.cudnn_enabled:
        xp = cuda.cupy
    else:
        xp = np

    with open(os.path.join(data_dir,'source_ids.pkl'),'rb') as f:source_ids=dill.load(f)
    with open(os.path.join(data_dir,'target_ids.pkl'),'rb') as f:target_ids=dill.load(f)
    with open(os.path.join(data_dir,'source_words.pkl'),'rb') as f:source_words=dill.load(f)
    with open(os.path.join(data_dir,'target_words.pkl'),'rb') as f:target_words=dill.load(f)

    args = ParseArg()

    model = Transformer(
        args.layer,
        min(len(source_ids), len(source_words)),
        min(len(target_ids), len(target_words)),
        args.unit,
        h=args.head,
        dropout=args.dropout,
        max_length=500,
        use_label_smoothing=args.use_label_smoothing,
        embed_position=args.embed_position)

    serializers.load_npz(os.path.join(data_dir,"best_model.npz"),model)

    
    # 分かち書きしてくれるオブジェクトを作成。
    tokenizer_obj = Tokenizer()


    return model,source_ids,target_words,tokenizer_obj