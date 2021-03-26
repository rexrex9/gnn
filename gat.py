import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.data import CoraGraphDataset
from dgl.nn import GATConv

class GAT( nn.Module ):
    def __init__(self,
                 g, #DGL的图对象
                 n_layers, #层数
                 in_feats, #输入特征维度
                 n_hidden, #隐层特征维度
                 n_classes, #类别数
                 heads, #多头注意力的数量
                 activation, #激活函数
                 in_drop, #输入特征的Dropout比例
                 at_drop, #注意力特征的Dropout比例
                 negative_slope, #注意力计算中Leaky ReLU的a值
                 ):
        super( GAT, self ).__init__( )
        self.g = g
        self.num_layers = n_layers
        self.activation = activation

        self.gat_layers = nn.ModuleList()

        self.gat_layers.append( GATConv(
            in_feats, n_hidden, heads[0],
            in_drop, at_drop, negative_slope, activation=self.activation ) )

        for l in range(1, n_layers):
            self.gat_layers.append( GATConv(
                n_hidden * heads[l-1], n_hidden, heads[l],
                in_drop, at_drop, negative_slope, activation=self.activation))

        self.gat_layers.append( GATConv(
            n_hidden * heads[-2], n_classes, heads[-1],
            in_drop, at_drop, negative_slope, activation=None) )

    def forward( self, inputs ):
        h = inputs
        for l in range( self.num_layers ):
            h = self.gat_layers[l]( self.g, h ).flatten( 1 )
        logits = self.gat_layers[-1]( self.g, h ).mean( 1 )
        return logits


def evaluate( model, features, labels, mask ):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def train( n_epochs = 100,
           lr = 5e-3,
           weight_decay = 5e-4,
           n_hidden = 16,
           n_layers = 1,
           activation = F.elu,
           n_heads = 3, #中间层多头注意力的数量
           n_out_heads = 1, #输出层多头注意力的数量
           feat_drop = 0.6,
           attn_drop = 0.6,
           negative_slope = 0.2):
    data = CoraGraphDataset()
    g=data[0]
    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    in_feats = features.shape[1]
    n_classes = data.num_labels
    heads = ([n_heads] * n_layers) + [n_out_heads]
    model = GAT( g,
                 n_layers,
                 in_feats,
                 n_hidden,
                 n_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope
                 )

    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam( model.parameters(),
                                 lr = lr,
                                 weight_decay = weight_decay)
    for epoch in range( n_epochs ):
        model.train()
        logits = model( features )
        loss = loss_fcn( logits[ train_mask ], labels[ train_mask ] )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = evaluate( model, features, labels, val_mask )
        print("Epoch {} | Loss {:.4f} | Accuracy {:.4f} "
              .format(epoch, loss.item(), acc ))
    print()
    acc = evaluate(model, features, labels, test_mask)
    print("Test accuracy {:.2%}".format(acc))

if __name__ == '__main__':
    train()