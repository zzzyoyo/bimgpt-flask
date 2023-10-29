import jieba
import jieba.posseg as pseg
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from torch import cuda
import json
import re
import os

device = 'cuda' if cuda.is_available() else 'cpu'
MAX_LEN = 256
BATCH_SIZE = 32


class MultiLabelDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.text = texts
        self.targets = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = self.text[index]
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }


class MultiLabelBERTModel(nn.Module):
    def __init__(self):
        super(MultiLabelBERTModel, self).__init__()
        self.bert = BertModel.from_pretrained(r'D:\bimgpt\d40_rulegeneration\Coarse&Fine-grained Classification\bert-chinese-base')
        ### New layers:
        self.linear1 = nn.Linear(768, 256)
        self.linear2 = nn.Linear(256, 32)  # 32个细粒度类型

    def forward(self, ids, mask):
        outputs = self.bert(
            ids,
            attention_mask=mask)
        linear1_output = self.linear1(outputs.pooler_output.view(-1, 768))
        linear2_output = self.linear2(linear1_output)

        return linear2_output


class MultiClassDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.text = texts
        self.targets = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = self.text[index]
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']  # 序列长度小于max_length时要用到，本场景大多数都是词语，所以会用到
        token_type_ids = inputs["token_type_ids"]  # 序列中有上下句的时候用到，本场景不需要

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.long)
        }


class SingleLabelBERTModel(nn.Module):
    def __init__(self):
        super(SingleLabelBERTModel, self).__init__()
        self.bert = BertModel.from_pretrained(r'D:\bimgpt\d40_rulegeneration\Coarse&Fine-grained Classification\bert-chinese-base')
        self.classifier = nn.Linear(768, 4)  # 粗粒度有4类

    def forward(self, ids, mask):
        outputs = self.bert(
            ids,
            attention_mask=mask)
        linear_output = self.classifier(outputs.pooler_output)
        return linear_output


def refine_words(word):
    refined_words = []
    word_list = jieba.lcut(word)
    new_word = ""
    for w in word_list:
        if w in ["等", "的", "以", "为", "为了", "由", "在", "对", "对于",  "和", "、", "与", "及", "以及", "及其", "如果", "那么", "否则", "且", "并且", "并", "或", "或者", "应", "不应", "应该", "应当", "必须", "宜", "不宜", "可", "不", "不可", "不得", "严禁", "等于", "不等于", "大于", "不大于", "小于", "不小于", "高于", "不高于", "低于", "不低于",
            "至少", "至多", "少于", "不少于", "多于", "不多于",
            "+", "-", "*", "/", ">", "<", ">=", "<=", "=", "!=", "有", "具有", "设置", "设有", "提供", "纳入",  "满足", "符合", "遵循", "依据", "根据", "按", "按照"]:
            if len(new_word) > 0:
                refined_words.append(new_word)
            refined_words.append(w)
            new_word = ""
        else:
            new_word += w
    if len(new_word) > 0:
        refined_words.append(new_word)
    return refined_words



def segment(resps):
    for pair in resps:
        resp = pair['response'].strip()
        start = 0
        current = 0
        words = []
        while current < len(resp):
            if resp[current] in ['\n', '\t', '\r', ' ', '，', '。', '；', '-', '、', '(', ')', '（', '）']:
                if current > start:
                    words += refine_words(resp[start:current])
                current += 1
                start = current
            else:
                current += 1
        if current > start:
            words += refine_words(resp[start:current])
        pair['words'] = words
    # print(resps)
    return resps


def single_label_validation(model, testing_loader):
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for data in testing_loader:
            ids = data['ids'].to(device)
            mask = data['mask'].to(device)
            token_type_ids = data['token_type_ids'].to(device)
            targets = data['targets'].to(device)
            outputs = model(ids, mask)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            # 多标签分类是把每个位置的值都映射到[0,1]区间，所以用sigmoid，而多分类除此之外还要求值的和为1，所以用softmax
            fin_outputs.extend(torch.softmax(outputs, dim=1).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets


def multi_label_validation(model, testing_loader):
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for data in testing_loader:
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)
            outputs = model(ids, mask)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets


def classify(dictionary, words):
    results_map = {
        "NounItems": {},
        "VerbItems": {},
        "AdItems": {},
        "OtherItems": {}
    }
    test_texts = []
    for word in words:
        # 先从字典里找
        if word in dictionary["NounItems"]:
            results_map["NounItems"][word] = dictionary["NounItems"][word]
            continue
        elif word in dictionary["VerbItems"]:
            results_map["VerbItems"][word] = dictionary["VerbItems"][word]
            continue
        elif word in dictionary["AdItems"]:
            results_map["AdItems"][word] = dictionary["AdItems"][word]
            continue
        elif word in dictionary["OtherItems"]:
            results_map["OtherItems"][word] = dictionary["OtherItems"][word]
            continue
        # 再用jieba分词来分roottag
        jieba.add_word(word)
        pseg_result = pseg.lcut(word)
        assert len(pseg_result) == 1
        word, flag = pseg_result[0]
        # print(word, '|', flag)
        if 'v' in flag:
            results_map["VerbItems"][word] = {
                "Name": word,
                "RootTag": "VERB",
                "SecondaryTag": flag,
                "FineTags": [],
                "DataMapping": None
            }
        elif 'a' in flag or 'd' in flag:
            results_map["AdItems"][word] = {
                "Name": word,
                "RootTag": "AD",
                "SecondaryTag": flag,
                "FineTags": [],
                "DataMapping": None
            }
        else:
            # 都拿去分类
            test_texts.append(word)

    print(test_texts)
    tokenizer = BertTokenizer.from_pretrained(r'D:\bimgpt\d40_rulegeneration\Coarse&Fine-grained Classification\bert-chinese-base')
    test_labels = [0 for i in range(len(test_texts))]  # 没有标签的测试集，随便生成一个长度和texts一样的即可
    testing_set = MultiClassDataset(test_texts, test_labels, tokenizer, MAX_LEN)
    test_params = {'batch_size': BATCH_SIZE,
                   'shuffle': False,
                   'num_workers': 0}
    testing_loader = DataLoader(testing_set, **test_params)
    # 粗粒度分类器
    model = SingleLabelBERTModel()
    model.to(device)
    model.load_state_dict(torch.load(r'D:\bimgpt\d40_rulegeneration\Coarse&Fine-grained Classification\multi_classify_model.pth'))
    coarse_outputs, _ = single_label_validation(model, testing_loader)
    coarse_outputs = np.argmax(coarse_outputs, axis=1)
    # 分类为建筑实体的数据集
    entities = []
    for i in range(len(coarse_outputs)):
        if coarse_outputs[i] == 0:
            entities.append(test_texts[i])
    entity_labels = [0 for i in range(len(entities))]
    testing_set = MultiLabelDataset(entities, entity_labels, tokenizer, MAX_LEN)
    testing_loader = DataLoader(testing_set, **test_params)
    # 细粒度分类器
    model = MultiLabelBERTModel()
    model.to(device)
    model.load_state_dict(torch.load(r'D:\bimgpt\d40_rulegeneration\Coarse&Fine-grained Classification\multilabel_classify_with_others_model.pth'))
    fine_outputs, _ = multi_label_validation(model, testing_loader)
    fine_outputs = np.array(fine_outputs) >= 0.6

    j = 0
    tag_map = {
        0: "ENT",
        1: "PROP",
        2: "PEOPLE",
        3: "OTHER"
    }
    with open(r'D:\bimgpt\d40_rulegeneration\Coarse&Fine-grained Classification\dataset\threshold=50,level=0\IndexToLabels.txt', 'r', encoding='utf-8') as f:
        labels_name = [line[line.index(':') + 1:].replace('\n', '') for line in f.readlines()]
    for i in range(len(test_texts)):
        if coarse_outputs[i] != 0:
            results_map["NounItems"][test_texts[i]] = {
                "Name": test_texts[i],
                "RootTag": "NOUN",
                "SecondaryTag": tag_map[coarse_outputs[i]],
                "FineTags": [],
                "DataMapping": None
            }
        else:
            labels = fine_outputs[j]
            j += 1
            names = []
            for n in range(32):
                if labels[n]:
                    names.append(labels_name[n])
            results_map["NounItems"][test_texts[i]] = {
                "Name": test_texts[i],
                "RootTag": "NOUN",
                "SecondaryTag": tag_map[coarse_outputs[i]],
                "FineTags": names,
                "DataMapping": None
            }
    return results_map