import random
import torch
from tqdm import tqdm #产生进度条的库
import numpy as np
import os
def readTriple(path,sep=None):
    with open(path,'r',encoding='utf-8') as f:
        for line in f.readlines():
            if sep:
                lines = line.strip().split(sep)
            else:
                lines=line.strip().split()
            if len(lines)!=3:continue
            yield lines

class Ml_100K():
    __BASE = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'ml-100k')
    KG=os.path.join(__BASE,'kg_index.tsv')
    RATING = os.path.join(__BASE,'rating_index.tsv')
    RATING5 = os.path.join(__BASE, 'rating_index_5.tsv')


def readKgData(path):
    print('读取知识图谱三元组...')
    entity_set,relation_set = set(),set()
    triples=[]
    for h, r, t in tqdm(readTriple(path)):
        entity_set.add(int(h))
        entity_set.add(int(t))
        relation_set.add(int(r))
        triples.append([int(h),int(r),int(t)])
    #返回实体集合列表，关系集合列表，与三元组列表
    return list(entity_set),list(relation_set),triples

def readRecData(path,test_ratio=0.2):
    print('读取用户评分三元组...')
    user_set,item_set=set(),set()
    triples=[]
    for u, i, r in tqdm(readTriple(path)):
        user_set.add(int(u))
        item_set.add(int(i))
        triples.append((int(u),int(i),int(r)))

    test_set = random.sample(triples,int(len(triples)*test_ratio))
    train_set = list(set(triples)-set(test_set))

    #返回用户集合列表，物品集合列表，与用户，物品，评分三元组列表
    return list(user_set),list(item_set),train_set,test_set

def setForTopKevaluation(testSet):
    all_testItems = set()
    user_items=dict()
    for u,v,r in testSet:
        all_testItems.add(v)
        if u not in user_items:
            user_items[u]={
                'pos':set(),
                'neg':set()
            }
        if r=='1':
            user_items[u]['pos'].add(v)
        else:
            user_items[u]['neg'].add(v)
    return all_testItems,user_items

def construct_kg(kgTriples):
    print('生成知识图谱索引图')
    kg = dict()
    for triple in kgTriples:
        head = triple[0]
        relation = triple[1]
        tail = triple[2]
        if head not in kg:
            kg[head] = []
        kg[head].append((tail, relation))
        if tail not in kg:
            kg[tail] = []
        kg[tail].append((head, relation))
    return kg

#根据kg邻接列表，得到实体邻接列表和关系邻接列表
def construct_adj(neighbor_sample_size, kg, entity_num):
    print('生成实体邻接列表和关系邻接列表')
    adj_entity = np.zeros([entity_num, neighbor_sample_size], dtype=np.int64)
    adj_relation = np.zeros([entity_num, neighbor_sample_size], dtype=np.int64)
    for entity in range(entity_num):
        neighbors = kg[entity]
        n_neighbors = len(neighbors)
        if n_neighbors >= neighbor_sample_size:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=neighbor_sample_size, replace=False)
        else:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=neighbor_sample_size, replace=True)
        adj_entity[entity] = np.array([neighbors[i][0] for i in sampled_indices])
        adj_relation[entity] = np.array([neighbors[i][1] for i in sampled_indices])

    return adj_entity, adj_relation


class DataIter():

    def __init__(self):
        pass

    def iter( self, recPairs, batchSize ):
        #传入评分三元组，知识图谱三元组，batch size
        for i in range( len(recPairs)//batchSize ):
            recDataSet = random.sample( recPairs, batchSize)
            yield torch.LongTensor( recDataSet)


if __name__ == '__main__':
    users, items, train_set, test_set = readRecData(Ml_100K.RATING)
    entitys, relations, kgTriples = readKgData(Ml_100K.KG)
    kg = construct_kg(kgTriples)
    adj_entity, adj_relation = construct_adj(5, kg, len(entitys))
    print(adj_entity, adj_relation)




