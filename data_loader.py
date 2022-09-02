import torch
from torch.utils.data import Dataset


class NERDataset(Dataset):

    def __init__(self, words, labels, word2id, label2id, unk_id):
        self.label2id = label2id
        self.word2id = word2id
        self.unk_id = unk_id
        self.dataset = self.preprocess(words, labels)


    def preprocess(self, words, labels):
        """文字转化成数字"""
        processed = []
        # 训练集和数据集全变成doc2vec向量格式

        for (word, label) in zip(words, labels):
            # 单词转化表如果失败的情况
            word_id = []
            for w_ in word:
                word_id.append(self.word2id.get(w_, self.unk_id))
            label_id = [self.label2id[l_] for l_ in label]
            processed.append((word_id, label_id))

        print (processed[:1])
        print ("-------- Process Done! --------")
        return processed

    def __getitem__(self, idx):
        word = self.dataset[idx][0]
        label = self.dataset[idx][1]
        return [word, label]

    def __len__(self):
        return len(self.dataset)

    def get_long_tensor(self, texts, labels, batch_size):

        token_len = max([len(x) for x in texts]) # 获取最长的
        text_tokens = torch.LongTensor(batch_size, token_len).fill_(0)
        label_tokens = torch.LongTensor(batch_size, token_len).fill_(0)
        mask_tokens = torch.ByteTensor(batch_size, token_len).fill_(0)

        for i, s in enumerate(zip(texts, labels)):
            text_tokens[i, :len(s[0])] = torch.LongTensor(s[0])
            label_tokens[i, :len(s[1])] = torch.LongTensor(s[1])
            mask_tokens[i, :len(s[0])] = torch.tensor([1] * len(s[0]), dtype=torch.uint8)

        return text_tokens, label_tokens, mask_tokens

    def collate_fn(self, batch):
        # 整理数据 batch
        texts = [x[0] for x in batch]
        labels = [x[1] for x in batch]
        lens = [len(x) for x in texts]
        batch_size = len(batch)

        input_ids, label_ids, input_mask = self.get_long_tensor(texts, labels, batch_size)

        return [input_ids, label_ids, input_mask, lens]