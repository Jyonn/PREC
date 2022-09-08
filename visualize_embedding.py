import argparse
import datetime

import numpy as np
import torch

from sklearn.manifold import TSNE


# nohup python visualize_embedding.py --type fuxi --path DCN_mind/MIND_x0_151efb4d/DCN_mind_base_008_457f38e9.model --exp vis.mind.news.e2e.csv > vis.mind.news.e2e.log &

def get_bert_embedding(path):
    d = torch.load(path)
    d = d['embedding_layer.embedding_layer.embedding_layers.nid.embedding_matrix.weight']
    return d.cpu().numpy()[1:-1]


def get_fuxi_embedding(path):
    d = torch.load(path)
    d = d['embedding_layer.embedding_layer.embedding_layers.nid.weight']
    return d.cpu().numpy()[1:-1]


parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str)
parser.add_argument('--path', type=str)
parser.add_argument('--export', type=str)

args = parser.parse_args()

if args.type == 'fuxi':
    embedding = get_fuxi_embedding(args.path)
else:
    embedding = get_bert_embedding(args.path)

print(datetime.datetime.now())
tsne = TSNE(n_components=2)
tsne.fit_transform(embedding)

print(datetime.datetime.now())
with open(args.export, 'w') as f:
    f.write('x\ty\n')
    for emb in tsne.embedding_:
        f.write('%.2f\t%.2f\n' % (emb[0], emb[1]))
