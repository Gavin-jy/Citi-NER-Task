from cgitb import text
import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from data_loader import NERDataset
from metric import f1_score, final_eval
from tqdm import tqdm

import numpy as np

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

def epoch_train(train_loader, model, optimizer, device, epoch, scheduler):
    # set model to training mode
    model.train()

    train_loss = 0.0
    for idx, batch_samples in enumerate(tqdm(train_loader)):
        x, y, mask, lens = batch_samples
        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)
        model.zero_grad()
        loss = model.compute_loss(x, mask, y)
        train_loss += loss.item()
        # 梯度反传
        loss.backward()
        # 优化更新
        optimizer.step()
        optimizer.zero_grad()
    scheduler.step()
    train_loss = float(train_loss) / len(train_loader)

    print("epoch: {}, train loss: {}".format(epoch, train_loss))

def train(train_loader, test_loader, word2id, tag2id, model, optimizer, device, scheduler):
    """训练模型 评估模型"""
    best_val_f1 = 0.0
    patience_counter = 0
    # start training
    save_df = pd.DataFrame({'epoch': [],
                            'loss': [],
                            'f1 score': []})
    for epoch in range(1, 301):
        epoch_train(train_loader, model, optimizer, device, epoch, scheduler)
        with torch.no_grad(): # 不再自动求导
            # dev loss calculation
            metric = dev(test_loader, word2id, tag2id, model, device)
            val_f1 = metric['f1']
            test_loss = metric['loss']
            print("epoch: {}, f1 score: {}, "
                            "dev loss: {}".format(epoch, val_f1, test_loss))

            df = pd.DataFrame({'epoch': [epoch],
                                'loss': [test_loss],
                                'f1 score': [val_f1]})
            
            save_df = save_df.append(df,ignore_index=True)
            # 轮数、f1分数、最终损失
            # 长时间进步小 停止
            improve_f1 = val_f1 - best_val_f1
            if improve_f1 > 1e-5: # 进步较大
                best_val_f1 = val_f1
                torch.save(model,'./save_model.pth')
                print("--------Save best model!--------")
                if improve_f1 < 0.0002:
                    patience_counter += 1
                else:
                    patience_counter = 0
            
            else: # 退步 或者 进步比较小
                patience_counter += 1
            # Early stopping and logging best f1
            if (patience_counter >= 10 and epoch > 5) or epoch == 300:
                print("Best val f1: {}".format(best_val_f1))
                break
    print("Training Finished!")
    save_df.to_csv('train_situation.csv',index=False)

def dev(data_loader, word2id, tag2id, model, device):
    """验证集"""
    model.eval()
    true_tags = []
    pred_tags = []
    test_losses = 0
    
    # 反转字典 
    id2word = {_idx: _word for _word, _idx in list(word2id.items())}
    id2tag = {_idx: _tag for _tag, _idx in list(tag2id.items())}
    
    for idx, batch_samples in enumerate(data_loader):
        sentences, labels, masks, lens = batch_samples
        # print("sentences{}".format(sentences))
        sentences = sentences.to(device)
        labels = labels.to(device)
        masks = masks.to(device)
        y_pred = model.forward(sentences)
        labels_pred = model.crf.decode(y_pred, mask=masks)
        # print("labels_pred{}".format(labels_pred))
        targets = [itag[:ilen] for itag, ilen in zip(labels.cpu().numpy(), lens)]
        true_tags.extend([[id2tag.get(idx) for idx in indices] for indices in targets])
        pred_tags.extend([[id2tag.get(idx) for idx in indices] for indices in labels_pred])
        # print(pred_tags)
        # print("pred_tags{}".format(labels_pred))
        # 计算损失
        test_loss = model.compute_loss(sentences, masks, labels)
        test_losses += test_loss
    
    # logging loss, f1 and report
    metrics = {}
    
    f1 = f1_score(true_tags, pred_tags)[0]
    metrics['f1'] = f1
    metrics['loss'] = float(test_losses) / len(data_loader)

    return metrics

def test(test_loader, word2id, tag2id, model, device):
    model = torch.load('save_model.pth')
    model.to(device)
    model.eval()
    true_tags = []
    pred_tags = []
    test_losses = 0
    
    # 反转字典 
    id2word = {_idx: _word for _word, _idx in list(word2id.items())}
    id2tag = {_idx: _tag for _tag, _idx in list(tag2id.items())}
   
    # 主要是把数字联合mask化成对应的文本数据 
    for idx, batch_samples in enumerate(test_loader):
        sentences, labels, masks, lens = batch_samples
        
        sentences = sentences.to(device)
        labels = labels.to(device)
        masks = masks.to(device)
        y_pred = model.forward(sentences)
        labels_pred = model.crf.decode(y_pred, mask=masks)
        targets = [itag[:ilen] for itag, ilen in zip(labels.cpu().numpy(), lens)]
        true_tags.extend([[id2tag.get(idx) for idx in indices] for indices in targets])
        pred_tags.extend([[id2tag.get(idx) for idx in indices] for indices in labels_pred])
        # 计算损失
        test_loss = model.compute_loss(sentences, masks, labels)
        test_losses += test_loss
    
    # logging loss, f1 and report
    metrics = {}
    
    f1_labels, f1 = final_eval(true_tags, pred_tags)
    metrics['f1_labels'] = f1_labels
    metrics['f1'] = f1
    metrics['loss'] = float(test_losses) / len(test_loader)
    return metrics