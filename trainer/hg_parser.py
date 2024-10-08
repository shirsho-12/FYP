# coding: utf-8

# ----------------------------------------------------------------
# Author:   Mouad Hakam (e1002601@nus.edu.sg)
# Date:     14/11/2023
# ----------------------------------------------------------------

import time
import json
import stanza
from tqdm import tqdm
from typing import List
from functools import lru_cache
from multiprocessing import Pool
from torch_geometric.data import HeteroData
import torch


from project.consts.constants import dicto


class ConstituencyNode(object):
    def __init__(self, cid, label, text, lids, tids, children=[], is_answer=False):
        self.cid = cid
        self.label = label
        self.text = text
        self.lids = lids
        self.tids = tids
        self.children = children
        self.is_answer = is_answer

    def __str__(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True)

    def __repr__(self):
        return self.__str__()

    @classmethod
    def iterate(cls, root):
        print(root)
        for child in root.children:
            ConstituencyNode.iterate(child)


class ConstituencyParser(object):
    def __init__(self, use_gpu: bool = True):
        self.pipelines = stanza.Pipeline(
            lang="en",
            processors="tokenize,pos,constituency",
            use_gpu=use_gpu,
            tokenize_batch_size=16,
            pos_batch_size=16,
        )

    @lru_cache(maxsize=64, typed=False)
    def get_sentences(self, doc: str) -> List:
        sentences = self.pipelines(doc).sentences
        return sentences

    def get_labels(self, answer: str) -> List:
        answer_sentences = self.get_sentences(answer)
        return answer_sentences[0].constituency.leaf_labels()


pid, cid = 0, 1000000000
R_cc, R_ct = list(), list()

cid = 0


def iterrate(tree, node_to_token_mapping, R_cc, R_ct, labels):
    global cid
    if tree.is_preterminal():

        tids = list()
        print("what am I looking for : ", tree.leaf_labels())
        mapped = False
        for i in range(len(node_to_token_mapping)):
            if tree.leaf_labels()[0] == node_to_token_mapping[i]:
                print("found it : ", node_to_token_mapping[i])
                mapped = True
                R_ct.append([cid, i])
                tids.append(i)
                break
        if not mapped:
            for i in range(len(node_to_token_mapping)):
                if "##" in node_to_token_mapping[i]:
                    node_to_token_mapping[i] = node_to_token_mapping[i][2:]
                if node_to_token_mapping[i] in tree.leaf_labels()[0]:
                    print("found it : ", node_to_token_mapping[i])
                    R_ct.append([cid, i])
                    tids.append(i)
                    continue
        leaf = ConstituencyNode(
            cid=cid, text=tree.leaf_labels()[0], label=tree.label, lids=[], tids=tids
        )

        labels[cid] = tree.label
        cid = cid + 1
        return leaf, R_cc, R_ct, labels
    else:
        labels1 = dict()
        leaves = list()
        for child in tree.children:
            leef, R_cc, R_ct, labe = iterrate(
                child, node_to_token_mapping, R_cc, R_ct, labels
            )
            leaves.append(leef)
            labels1.update(labe)
        text = ""
        lids = list()
        tids = list()
        for leaf in leaves:
            text = text + leaf.text
            lids.append(leaf.cid)
            tids.extend(leaf.tids)
            R_cc.append([cid, leaf.cid])
        labels1[cid] = tree.label
        con = ConstituencyNode(
            cid=cid, text=text, label=tree.label, lids=lids, tids=tids
        )
        cid = cid + 1
        return con, R_cc, R_ct, labels1


from transformers import BertTokenizer

tz = BertTokenizer.from_pretrained("bert-base-cased")

# Unit test
if __name__ == "__main__":
    # 1. Constituency Parser
    cp = ConstituencyParser(use_gpu=True)

    def process(text, p_index):
        result = cp.get_sentences(text)
        return result[0].constituency

    for i in tqdm(range(100)):
        # text = "The Norman's dynasty had a major political, cultural and military impact on medieval Europe and even the Near East."
        text = "I like potatoes"
        results = process(text, 0)

    tokenized = tz.tokenize(text)
    ids = tz.convert_tokens_to_ids(tokenized)

    ans = cp.get_labels(text)

    constituents = cp.get_sentences(text)

    child_nodes, lids, tids = list(), list(), list()
    for constituent in constituents:
        # print(constituent.constituency)
        data = HeteroData()
        con = constituent.constituency
        m, c, t, lab = iterrate(con, tokenized, list(), list(), dict())

        pre = []
        for key in lab:
            sam = torch.zeros(60)
            number = dicto[lab[key]]
            sam[number] = 1
            pre.append(sam)
        resu = torch.stack(pre, 0)
        data["constituent"].node_id = torch.arange(len(lab))
        data["tokens"].node_id = torch.arange(len(m.tids))
        data["tokens"].x = torch.reshape(torch.tensor(ids), (len(ids), 1))
        data["constituent"].x = resu

        data["constituent", "connect", "constituent"] = torch.tensor(c)
        data["constituent", "connect", "token"] = torch.tensor(t)
        print(data)
