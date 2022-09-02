import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from torchcrf import CRF

torch.manual_seed(1)



def prepare_sequence(seq, to_ix):
    """
    转化词成为对应的向量 （根据准备的词表）
    """
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim # 嵌入维度
        self.hidden_dim = hidden_dim # 隐藏层维度
        self.vocab_size = vocab_size # 词表大小 
        self.tag2ix = tag_to_ix # 标志表
        self.tagset_size = len(tag_to_ix)  # 目标取值范围大小

        self.embedding = nn.Embedding(vocab_size, embedding_dim) # 嵌入层
        
        self.lstm = nn.LSTM(
            input_size = embedding_dim, # 32
            hidden_size = hidden_dim // 2, # 16
            num_layers=2, 
            dropout = 0.5,
            batch_first = True,
            bidirectional=True
        )

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size) # 全连接
        self.crf = CRF(self.tagset_size, batch_first = True)

    def forward(self, input_id):
        # bi-lstm 的得分
        embeddings = self.embedding(input_id)
        sequence_output, _ = self.lstm(embeddings)
        tag_scores = self.hidden2tag(sequence_output)
        return tag_scores

    def compute_loss(self, input_id, input_mask, input_tags):
        tag_scores = self.forward(input_id)
        loss = - self.crf(tag_scores, input_tags, input_mask)
        return loss
    
    def decode(self, input_id):
        tag_scores = self.forward(input_id)
        pre = self.crf.decode(tag_scores)
        return pre



