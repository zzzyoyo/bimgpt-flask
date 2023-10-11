import jieba
import jieba.posseg as pseg
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from torch import cuda
import json
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


def segment(file_content: str):
    return pseg.cut(file_content)


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


def classify(words):
    results_map = {
        "NounItems": {},
        "VerbItems": {},
        "AdItems": {},
        "OtherItems": {}
    }
    test_texts = []
    for word, flag in words:
        if 'v' in flag:
            results_map["VerbItems"][word] = {"flag": flag}
        elif 'a' in flag or 'd' in flag:
            results_map["AdItems"][word] = {"flag": flag}
        else:
            # 都拿去分类
            test_texts.append(word)

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
    with open('dataset/threshold=50,level=0/IndexToLabels.txt', 'r', encoding='utf-8') as f:
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