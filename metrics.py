from sklearn.metrics import roc_auc_score
import numpy as np

NUM_RECS_RANGE = 20

def roc_auc(true_y, predicted_y):

    return roc_auc_score(true_y, predicted_y)


def hit_rate(hit_vec_np):
    hitrate = []
    for num_recs in range(5, NUM_RECS_RANGE + 1):
        if np.sum(hit_vec_np[: num_recs]) > 0:
            hitrate.append(1)
        else:
            hitrate.append(0)

    return np.array(hitrate)

def ndcg(hit_vec_np):
    ndcg = []
    for num_recs in range(5, NUM_RECS_RANGE + 1):
        hit = np.array(hit_vec_np[: num_recs], dtype=np.int)
        hit = hit.reshape(1, -1)
        ndcg.append(np.sum(hit) / (np.log2(np.argmax(hit) + 2)))

    return np.array(ndcg)