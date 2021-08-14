import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm #产生进度条
import dataloader4kg
from sklearn.metrics import roc_auc_score,precision_score,recall_score,accuracy_score

class KGCN( nn.Module ):

    def __init__( self, n_users, n_entitys, n_relations,
                  e_dim,  adj_entity, adj_relation, n_neighbors,
                  aggregator_method = 'sum',
                  act_method = F.relu, drop_rate=0.5):
        super( KGCN, self ).__init__()

        self.e_dim = e_dim  # 特征向量维度
        self.aggregator_method = aggregator_method #消息聚合方法
        self.n_neighbors = n_neighbors #邻居的数量
        self.user_embedding = nn.Embedding( n_users, e_dim, max_norm = 1 )
        self.entity_embedding = nn.Embedding( n_entitys, e_dim, max_norm = 1)
        self.relation_embedding = nn.Embedding( n_relations, e_dim, max_norm = 1)

        self.adj_entity = adj_entity #节点的邻接列表
        self.adj_relation = adj_relation #关系的邻接列表

        #线性层
        self.linear_layer = nn.Linear(
                in_features = self.e_dim * 2 if self.aggregator_method == 'concat' else self.e_dim,
                out_features = self.e_dim,
                bias = True)

        self.act = act_method #激活函数
        self.drop_rate = drop_rate #drop out 的比率

    def forward(self, users, items, is_evaluate = False):
        neighbor_entitys, neighbor_relations = self.get_neighbors( items )
        user_embeddings = self.user_embedding( users)
        item_embeddings = self.entity_embedding( items )

        #得到v波浪线
        neighbor_vectors = self.__get_neighbor_vectors( neighbor_entitys, neighbor_relations, user_embeddings )

        out_item_embeddings = self.aggregator( item_embeddings, neighbor_vectors,is_evaluate)

        out = torch.sigmoid( torch.sum( user_embeddings * out_item_embeddings, axis = -1 ) )

        return out

    def get_neighbors( self, items ):#得到邻居的节点embedding,和关系embedding
        #[[1,2,3,4,5],[2,1,3,4,5]...[]]#总共batchsize个n_neigbor的id
        entity_ids = [ self.adj_entity[item] for item in items ]
        relation_ids = [ self.adj_relation[item] for item in items ]
        neighbor_entities = [ torch.unsqueeze(self.entity_embedding(torch.LongTensor(one_ids)),0) for one_ids in entity_ids]
        neighbor_relations = [ torch.unsqueeze(self.relation_embedding(torch.LongTensor(one_ids)),0) for one_ids in relation_ids]
        # [batch_size, n_neighbor, dim]
        neighbor_entities = torch.cat( neighbor_entities, dim=0 )
        neighbor_relations = torch.cat( neighbor_relations, dim=0 )

        return neighbor_entities, neighbor_relations

    #得到v波浪线
    def __get_neighbor_vectors(self, neighbor_entitys, neighbor_relations, user_embeddings):
        # [batch_size, n_neighbor, dim]
        user_embeddings = torch.cat([torch.unsqueeze(user_embeddings,1) for _ in range(self.n_neighbors)],dim=1)
        # [batch_size, n_neighbor]
        user_relation_scores = torch.sum(user_embeddings * neighbor_relations, axis=2)
        # [batch_size, n_neighbor]
        user_relation_scores_normalized = F.softmax(user_relation_scores, dim=-1)
        # [batch_size, n_neighbor, 1]
        user_relation_scores_normalized = torch.unsqueeze(user_relation_scores_normalized, 2)
        # [batch_size, dim]
        neighbor_vectors = torch.sum(user_relation_scores_normalized * neighbor_entitys, axis=1)
        return neighbor_vectors

    #经过进一步的聚合与线性层得到v
    def aggregator(self,item_embeddings, neighbor_vectors, is_evaluate):
        # [batch_size, dim]
        if self.aggregator_method == 'sum':
            output = item_embeddings + neighbor_vectors
        elif self.aggregator_method == 'concat':
            # [batch_size, dim * 2]
            output = torch.cat([item_embeddings, neighbor_vectors], axis=-1)
        else:#neighbor
            output = neighbor_vectors

        if not is_evaluate:
            output = F.dropout(output, self.drop_rate)
        # [batch_size, dim]
        output = self.linear_layer(output)
        return self.act(output)

#验证
def do_evaluate( model, testSet ):
    testSet = torch.LongTensor(testSet)
    model.eval()
    with torch.no_grad():
        user_ids = testSet[:, 0]
        item_ids = testSet[:, 1]
        labels = testSet[:, 2]
        logits = model( user_ids, item_ids, True )
        predictions = [1 if i >= 0.5 else 0 for i in logits]
        p = precision_score(y_true = labels, y_pred = predictions)
        r = recall_score(y_true = labels, y_pred = predictions)
        acc = accuracy_score(labels, y_pred = predictions)
        return p,r,acc

def train( epochs, batchSize, lr,
           n_users, n_entitys, n_relations,
           adj_entity, adj_relation,
           train_set, test_set,
           n_neighbors,
           aggregator_method = 'sum',
           act_method = F.relu, drop_rate = 0.5, weight_decay=5e-4
         ):

    model = KGCN( n_users, n_entitys, n_relations,
                  10, adj_entity, adj_relation,
                  n_neighbors = n_neighbors,
                  aggregator_method = aggregator_method,
                  act_method = act_method,
                  drop_rate = drop_rate )
    optimizer = torch.optim.Adam( model.parameters(), lr = lr, weight_decay = weight_decay )
    loss_fcn = nn.BCELoss()
    dataIter = dataloader4kg.DataIter()
    print(len(train_set)//batchSize)

    for epoch in range( epochs ):
        total_loss = 0.0
        for datas in tqdm( dataIter.iter( train_set, batchSize = batchSize ) ):
            user_ids = datas[:, 0]
            item_ids = datas[:, 1]
            labels = datas[:, 2]
            logits = model.forward( user_ids, item_ids )
            loss = loss_fcn( logits, labels.float() )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        p, r, acc = do_evaluate(model,test_set)
        print("Epoch {} | Loss {:.4f} | Precision {:.4f} | Recall {:.4f} | Accuracy {:.4f} "
                  .format(epoch, total_loss/(len(train_set)//batchSize), p, r, acc))



if __name__ == '__main__':
    n_neighbors = 10

    users, items, train_set, test_set = dataloader4kg.readRecData( dataloader4kg.Ml_100K.RATING )
    entitys, relations, kgTriples = dataloader4kg.readKgData( dataloader4kg.Ml_100K.KG )
    adj_kg = dataloader4kg.construct_kg( kgTriples )
    adj_entity, adj_relation = dataloader4kg.construct_adj( n_neighbors, adj_kg, len( entitys ) )

    train( epochs = 10, batchSize = 1024, lr = 0.01,
           n_users = max( users ) + 1, n_entitys = max( entitys ) + 1,
           n_relations = max( relations ) + 1, adj_entity = adj_entity,
           adj_relation = adj_relation, train_set = train_set,
           test_set = test_set, n_neighbors = n_neighbors,
           aggregator_method = 'sum', act_method = F.relu, drop_rate = 0.5 )



