from flask import Flask
from flask_cors import CORS
import torch
import numpy as np
import requests
import numpy as np
import requests
from data_loader import NERDataset
from torch.utils.data import DataLoader
from torchcrf import CRF

app = Flask(__name__)  # 如果是单独应用可以使用__name__，如果是module则用模块名
# Flask还有其他参数https://blog.csdn.net/YZL40514131/article/details/122730037
CORS(app, supports_credentials=True)  # 解决跨域

# 序列字典 
tag2id = {
    "O": 0, 
    "B-NAME": 1, 
    "I-NAME": 2,
    "B-NOTIONAL": 4, 
    "I-NOTIONAL": 5,
    "B-TICKER": 6, 
    "I-TICKER": 7,
    '[CLS]': 10,
    '[SEP]': 11
}

id2tag = {_idx: _tag for _tag, _idx in list(tag2id.items())}

@app.route('/123')
def hello():
    return "hello world"

@app.route('/nlp/<string>')
def ner_function(string):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #print (device)
    #device = 'cpu'
    model = torch.load('save_model.pth')
    #print(model)
    model.to(device)
    word2ix = np.load('vocab.npy', allow_pickle='TRUE').item() # 加载词典
    pred_tags = [] # 

    split_word_list = string.split()
    low_word_list = [_t.lower() for _t in split_word_list] # 单词列表

    # sents = preprocess(word_list, word2ix) # 转化得来的向量
    label_list = []
    for i in range(0,len(low_word_list)):
        label_list.append('O')
    low_word_list = [low_word_list] 
    label_list = [label_list]
    dataset = NERDataset(low_word_list, label_list, word2ix, tag2id, len(word2ix)-1)
    
    loader = DataLoader(dataset, batch_size=1,
                            shuffle=True, collate_fn=dataset.collate_fn)

    for idx, batch_samples in enumerate(loader):
        sentences, labels, masks, lens = batch_samples
        sentences = sentences.to(device)
        labels = labels.to(device)
        masks = masks.to(device)
        y_pred = model.forward(sentences)
        labels_pred = model.crf.decode(y_pred, mask=masks)
        pred_tags.extend([[id2tag.get(idx) for idx in indices] for indices in labels_pred])

    # labels_pred = model(sents)
    
    print(pred_tags)
    
    word_list = [[_t for _t in split_word_list]]
    results = entity(word_list, pred_tags) # 找到对应的实体

    return results

def preprocess(seq, to_ix):
    idxs = []
    for w in seq:
        if w in to_ix.keys(): 
            idxs.append(to_ix[w])
        else:
            idxs.append(len(to_ix))
    return torch.tensor(idxs, dtype=torch.long)

def entity(word_list, labels):
    # 变成实体返回
    """
        输入是字符
    """
    results = []
    for i in range(0,len(word_list[0])):
        if labels[0][i][0] == 'B':
            word = word_list[0][i] 
            for j in range(i+1,len(word_list[0])):
                if labels[0][j][0] == 'I':
                    word += ' ' + word_list[0][j]
                else:
                    break
            label = labels[0][i].split('-')[1]

            results.append([word,label])

    return results
if __name__ == '__main__':
    # app.run()
    app.run(host='0.0.0.0')