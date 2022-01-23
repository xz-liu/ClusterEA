import torch
import torch.nn.functional as F


def align_loss(align_input, embedding, gamma, node_size, device):
    def squared_dist(x):
        A, B = x
        row_norms_A = torch.sum(torch.square(A), dim=1)
        row_norms_A = torch.reshape(row_norms_A, [-1, 1])  # Column vector.
        row_norms_B = torch.sum(torch.square(B), dim=1)
        row_norms_B = torch.reshape(row_norms_B, [1, -1])  # Row vector.
        # may not work
        return row_norms_A + row_norms_B - 2 * torch.matmul(A, torch.transpose(B, 0, 1))

    # modified
    left = torch.tensor(align_input[:, 0])
    right = torch.tensor(align_input[:, 1])
    l_emb = embedding[left]
    r_emb = embedding[right]
    pos_dis = torch.sum(torch.square(l_emb - r_emb), dim=-1, keepdim=True)
    r_neg_dis = squared_dist([r_emb, embedding])
    l_neg_dis = squared_dist([l_emb, embedding])

    l_loss = pos_dis - l_neg_dis + gamma
    l_loss = l_loss * (1 - F.one_hot(left, num_classes=node_size) - F.one_hot(right, num_classes=node_size)).to(device)

    r_loss = pos_dis - r_neg_dis + gamma
    r_loss = r_loss * (1 - F.one_hot(left, num_classes=node_size) - F.one_hot(right, num_classes=node_size)).to(device)
    # modified
    with torch.no_grad():
        r_mean = torch.mean(r_loss, dim=-1, keepdim=True)
        r_std = torch.std(r_loss, dim=-1, keepdim=True)
        r_loss.data = (r_loss.data - r_mean) / r_std
        l_mean = torch.mean(l_loss, dim=-1, keepdim=True)
        l_std = torch.std(l_loss, dim=-1, keepdim=True)
        l_loss.data = (l_loss.data - l_mean) / l_std
        # l_loss = (l_loss - torch.mean(l_loss, dim=-1, keepdim=True).detach()) / torch.std(
        #     l_loss, dim=-1, keepdim=True).detach()

    lamb, tau = 30, 10
    # l_loss = K.log(K.sum(K.exp(lamb * l_loss + tau), axis=-1))
    # r_loss = K.log(K.sum(K.exp(lamb * r_loss + tau), axis=-1))
    # l_loss = K.logsumexp(lamb * l_loss + tau, axis=-1)
    # r_loss = K.logsumexp(lamb * r_loss + tau, axis=-1)
    l_loss = torch.logsumexp(lamb * l_loss + tau, dim=-1)
    r_loss = torch.logsumexp(lamb * r_loss + tau, dim=-1)
    return torch.mean(l_loss + r_loss)
