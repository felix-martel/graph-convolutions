import torch

import fb15k
from rgcn import RGCN
from utils import graph, tensor

LR = 0.001
EPOCHS = 5
N_CLASSES = 10
N_BASIS_FUNCTIONS = 20

#
# Loading data
#

train, val, test = fb15k.load("train", "valid", "test")
print(f"{len(train)} training triples found.")

T, e2c, e2i, r2i = graph.build_adjacency_tensor(train)
n_relations = len(T)
n_entities = T[0].shape[0]
print(f"{n_entities} entities and {n_relations} relations found.")

# Supervised setting: each entity has a class. Here we build the ground truth, that is the expected output tensor
# Give a unique identifier to each class
classes = {c: i for i, c in enumerate(set(e2c))}
n_classes = len(classes)
y_true = [classes[c] for c in e2c]
y_true = torch.LongTensor(y_true)
print(f"{n_classes} distinct classes found.")


#
# Model definition
#

rgcn = RGCN(T,
            n_classes=n_classes,
            hidden_sizes=[64, 32, 16],
            n_basis=N_BASIS_FUNCTIONS
            )

print(rgcn)

#
# Training params
#

optim = torch.optim.Adam(rgcn.parameters(recurse=True), lr=LR)
cross_entropy = torch.nn.CrossEntropyLoss()
# We're in the featureless setting, so each entity is one-hot encoded, hence
# the input data is simply the identity matrix of dim N_entities x N_entities
I = tensor.sparse_eye(n_entities)

#
# Training
#

for i in range(EPOCHS):
    print(f"Step {i+1}/{EPOCHS}")
    optim.zero_grad()
    y_pred = rgcn(I)
    loss = cross_entropy(y_pred, y_true)
    loss.backward()
    optim.step()



