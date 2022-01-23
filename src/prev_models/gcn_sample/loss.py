import torch
import torch.nn.functional as F

from utils_largeea import norm_process


def marginLossGCN(pos_1, pos_2, neg_1, neg_2, margin=3):
    A = torch.norm(pos_1 - pos_2, p=1, dim=1, keepdim=False)
    B = torch.norm(neg_1 - neg_2, p=1, dim=1, keepdim=False)
    C = torch.norm(pos_1 - neg_2, p=1, dim=1, keepdim=False)
    D = torch.norm(pos_2 - neg_1, p=1, dim=1, keepdim=False)
    loss = sum(F.relu(A + margin - B)) + sum(F.relu(A + margin - C)) + sum(F.relu(A + margin - D))
    return loss


def loss_cosine(emb1, emb2):
    return 1 - torch.cosine_similarity(emb1, emb2, dim=1)


def loss_l1(emb1, emb2):
    return torch.norm(emb1 - emb2, p=1, dim=1, keepdim=False)


def loss_l2(emb1, emb2):
    return torch.norm(emb1 - emb2, p=2, dim=1, keepdim=False)


def marginLossRREA(pos_1, pos_2, neg_1, neg_2, margin=3):
    # seed_1 = links.T[0]
    # seed_2 = links.T[1]
    # train_emb_1 = embedding1[seed_1]
    # train_emb_2 = embedding2[seed_2]
    # # A = torch.norm(train_emb_1 - train_emb_2, p=1, dim=1, keepdim=False)
    # neg_seed_1 = np.random.randint(0, total, size=len(links))
    # neg_seed_2 = np.random.randint(0, total, size=len(links))
    # neg_emb_1 = embedding1[neg_seed_1]
    # neg_emb_2 = embedding2[neg_seed_2]
    # B = torch.norm(neg_emb_1 - neg_emb_2, p=1, dim=1, keepdim=False)
    # C = torch.norm(train_emb_1 - neg_emb_2, p=1, dim=1, keepdim=False)
    # D = torch.norm(train_emb_2 - neg_emb_1, p=1, dim=1, keepdim=False)
    # loss = sum(F.relu(A + margin - B)) + sum(F.relu(A + margin - C)) + sum(F.relu(A + margin - D))

    loss = torch.sum(torch.relu(margin + loss_l1(pos_1, pos_2) - loss_l1(pos_1, neg_2))
                     + torch.relu(margin + loss_l1(pos_1, pos_2) - loss_l1(neg_1, pos_2)))
    return loss
