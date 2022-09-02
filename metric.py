import os

LABELS = ['NAME','NOTIONAL','TICKER']

def get_entities(seq):
    """
    Gets entities from sequence.
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        get_entities(seq)
        [('PER', 0, 1), ('LOC', 3, 3)]
    """
    if any(isinstance(s, list) for s in seq):
        # 拆分嵌套的列表，将每层的后面加一个'0'然后变成一维
        seq = [item for sublist in seq for item in sublist + ['O']]

    prev_tag = 'O'
    prev_type = ''
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq + ['O']): # 枚举整个列表
        
        # 获得标签类型
        tag = chunk[0]
        # 获得实体类型
        type_ = chunk.split('-')[-1]
        
        if end_of_chunk(prev_tag, tag, prev_type, type_):
            chunks.append((prev_type, begin_offset, i - 1))
        if start_of_chunk(prev_tag, tag, prev_type, type_):
            begin_offset = i
        prev_tag = tag
        prev_type = type_

    return chunks


def end_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk ended between the previous and current word.
    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.
    Returns:
        chunk_end: boolean.
    """
    chunk_end = False

    # pred_label中可能出现这种情形
    if prev_tag == 'B' and tag == 'B':
        chunk_end = True
    if prev_tag == 'B' and tag == 'O':
        chunk_end = True
    if prev_tag == 'I' and tag == 'B':
        chunk_end = True
    if prev_tag == 'I' and tag == 'O':
        chunk_end = True
    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk started between the previous and current word.
    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.
    Returns:
        chunk_start: boolean.
    """
    chunk_start = False

    if tag == 'B':
        chunk_start = True

    if prev_tag == 'O' and tag == 'I':
        chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    return chunk_start

def f1_score(y_true, y_pred):
    """计算F1分数
        y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        f1_score(y_true, y_pred)
        0.50
    """
    true_entities = set(get_entities(y_true))
    pred_entities = set(get_entities(y_pred))
    # print(true_entities)
    # & 操作获得实体预测正确的部分
    # 取长度获得预测正确的个数
    nb_correct = len(true_entities & pred_entities)
    
    nb_pred = len(pred_entities)
    nb_true = len(true_entities)

    # 准确 = 正确预测/预测的个数 预测的准确程度
    p = nb_correct / nb_pred if nb_pred > 0 else 0 
    # 召回 = 正确预测/真实的个数 预测的寻找正样本能力
    r = nb_correct / nb_true if nb_true > 0 else 0 

    score = 2 * p * r / (p + r) if p + r > 0 else 0

    return score, p, r 

def final_eval(y_true, y_pred):

    true_entities = set(get_entities(y_true))
    pred_entities = set(get_entities(y_pred))
    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)
    nb_true = len(true_entities)

    p = nb_correct / nb_pred if nb_pred > 0 else 0
    r = nb_correct / nb_true if nb_true > 0 else 0
    score = 2 * p * r / (p + r) if p + r > 0 else 0
    
    f_score = {}
    
    for label in LABELS:
        true_entities_label = set()
        pred_entities_label = set()
        for t in true_entities:
            if t[0] == label:
                true_entities_label.add(t)
        for p in pred_entities:
            if p[0] == label:
                pred_entities_label.add(p)
        nb_correct_label = len(true_entities_label & pred_entities_label)
        nb_pred_label = len(pred_entities_label)
        nb_true_label = len(true_entities_label)

        p_label = nb_correct_label / nb_pred_label if nb_pred_label > 0 else 0
        r_label = nb_correct_label / nb_true_label if nb_true_label > 0 else 0
        score_label = 2 * p_label * r_label / (p_label + r_label) if p_label + r_label > 0 else 0
        f_score[label] = score_label
    return f_score, score
