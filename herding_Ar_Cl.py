import numpy as np
import torch
import time
from sklearn.metrics import f1_score
from model import LogisticRegression
import logging
import random
import torch
import torch.backends.cudnn as cudnn
import datetime
from sklearn.metrics import log_loss

import warnings
warnings.filterwarnings("ignore")

tsk = 'DomainNet/s2r' # 'Ar2Cl'
    
now = datetime.datetime.now()
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# File handler
file_handler = logging.FileHandler("/opt/kcutp/mbh_ui/active_learning/Category_Aware_DA/log/" + tsk + "/herding_" + timestamp + ".log",  mode='w')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(message)s'))

logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info("Session started")
    

def train_and_eval(train_x, train_y, val_x, val_y, one_idx, C):
    model = LogisticRegression(C)
    model.fit(train_x, train_y)
    
    # #print loss for ablation study
    # logger.info(f"Len train_x: {len(train_x)}")
    # val_prob = model.model.predict(val_x)
    # valid_loss_wo_reg = log_loss(val_y, val_prob)
    # logger.info(f"Valid accuracy: {model.model.score(val_x[one_idx], val_y[one_idx])}")
    # logger.info(f"Valid loss: {valid_loss_wo_reg}")
    
    return model.model.score(val_x, val_y), model.model.score(val_x[one_idx], val_y[one_idx]), f1_score(model.model.predict(val_x), val_y, average='binary'), model

def update_datasets(train_x, val_x, train_y, val_y, q_idxs):
    # query new dataset and retrain
    train_x_new = np.concatenate((train_x, val_x[q_idxs]))
    train_y_new = np.concatenate((train_y, val_y[q_idxs]))

    valid_x = val_x[q_idxs]
    valid_y = val_y[q_idxs]

    val_x_new = np.delete(val_x, q_idxs, axis=0)
    val_y_new = np.delete(val_y, q_idxs, axis=0)
    
    return  train_x_new, val_x_new, train_y_new, val_y_new, valid_x, valid_y

def retrain_model(train_x_new, train_y_new, val_x, val_y, one_idx, C, weights=None):
    model = LogisticRegression(C)
    model.fit(train_x_new, train_y_new, sample_weight=weights)
    
    # #print loss for ablation study
    # logger.info(f"Len train_x_new: {len(train_x_new)}")
    # val_prob = model.model.predict(val_x)
    # valid_loss_wo_reg = log_loss(val_y, val_prob)
    # logger.info(f"Valid accuracy: {model.model.score(val_x[one_idx], val_y[one_idx])}")
    # logger.info(f"Valid loss: {valid_loss_wo_reg}")
    
    return model.model.score(val_x, val_y), model.model.score(val_x[one_idx], val_y[one_idx]), f1_score(model.model.predict(val_x), val_y, average='binary'), model

def compute_norm(x1, x2, device, batch_size=512):
    x1, x2 = x1.unsqueeze(0).to(device), x2.unsqueeze(0).to(device) # 1 x n x d, 1 x n' x d
    dist_matrix = []
    batch_round = x2.shape[1] // batch_size + int(x2.shape[1] % batch_size > 0)
    for i in range(batch_round):
        # distance comparisons are done in batches to reduce memory consumption
        x2_subset = x2[:, i * batch_size: (i + 1) * batch_size]
        dist = torch.cdist(x1, x2_subset, p=2.0)

        dist_matrix.append(dist.cpu())
        del dist

    dist_matrix = torch.cat(dist_matrix, dim=-1).squeeze(0)
    return dist_matrix

def uncertainty(probs):
    entropy = []
    for p in probs:
        p = np.clip(p, 1e-12, 1 - 1e-12)  # tránh log(0)
        entropy.append(-p * np.log2(p) - (1 - p) * np.log2(1 - p))
    return torch.tensor(entropy)

class RBFKernel(object):
    def __init__(self, device):
        self.device = device

    def compute_kernel(self, x1, x2, h=0.6, batch_size=512):
        norm = compute_norm(x1, x2, self.device, batch_size=batch_size)
        k = torch.exp(-1.0 * (norm / h) ** 2)
        return k
    
def herding(model, train_x_new, val_x_new, train_y_new, batch_size=2048, budget=0):
    kernel = RBFKernel('cuda')
    
    train_x_t = torch.tensor(train_x_new).to('cuda')
    train_y_t = torch.tensor(train_y_new).to('cuda').view(-1)
    
    pos_mask = train_y_t == 1
    pos_x = train_x_t[pos_mask]
    val_x_t = torch.tensor(val_x_new).to('cuda')
    all_x_t = torch.cat((pos_x, val_x_t), dim=0)
    
    prob = model.model.predict_proba(all_x_t.cpu())[:, 1]
    uncertainties = uncertainty(prob)
    uncertainties[:len(pos_x)] = 0.
    uncertainties = uncertainties.reshape(1, -1)
    # uncertainties = torch.ones((1, len(all_x_t)))
    
    # scores for target pool
    k_all = kernel.compute_kernel(all_x_t, all_x_t, batch_size=batch_size)
    k_la = kernel.compute_kernel(pos_x, all_x_t, batch_size=batch_size)
    
    max_embedding = k_la.max(dim=0, keepdim=True).values

    selected = []
    for i in range(budget):
        start = time.time()
        updated_max_embedding = (k_all - max_embedding) # N x N
        updated_max_embedding[updated_max_embedding < 0] = 0.
        mean_max_embedding = (uncertainties*updated_max_embedding).mean(dim=-1) # N

        # select a point from u
        mean_max_embedding[:pos_x.shape[0]] = -np.inf
        mean_max_embedding[selected] = -np.inf
        selected_index = torch.argmax(mean_max_embedding)
        selected.append(selected_index.item())

        max_embedding = updated_max_embedding[selected_index].unsqueeze(0) + max_embedding
        # print("Done one selection with time:", time.time() - start)

    selected = [(x-pos_x.shape[0]) for x in selected]
    # tar_score = k_tar_pos.mean(dim=1)
    # # tar_score = k_tar_pos.mean(dim=1) - lambda_neg * k_tar_neg.mean(dim=1)
    
    # prob_pos = model.model.predict_proba(val_x_new)[:, 1]
    # # final_score = alpha * tar_score + (1.0 - alpha) * prob_pos
    # final_score = k_la.mean(dim=1)
    return torch.tensor(selected)
    
def active_learning(train_x, train_y, val_x, val_y, one):

    #enhance the performance for a specific category
    train_y = np.where(train_y == one, 1, 0)
    val_y = np.where(val_y == one, 1, 0)
    one_train = train_y==1
    one_idx = val_y==1
    
    L2_WEIGHT = 1e-3
    C = 1 / (train_x[0].shape[0] * L2_WEIGHT)
    
    src_acc = []
    acc, acc_one, f1 = [], [], []
    aa, sa, fa, model = train_and_eval(train_x, train_y, val_x, val_y, one_idx, C)
    ori_pred = model.model.predict(val_x[one_idx])
    ori_one = sa
    
    src_acc.append(model.model.score(train_x[one_train], train_y[one_train]))
    
    # n = int(val_x.shape[0]*0.01)
    n = 100
    print("Budget: ", n)

    pred_ones = (model.model.predict(val_x)==1).astype(int)
    one_ratio = 1.0 
    qones = min(int(n*one_ratio), pred_ones.sum())

    if qones == 0:
        one_idxs = np.argpartition(model.model.predict_proba(val_x)[:, 1], -int(n*one_ratio))[-int(n*one_ratio):]
    else:
        one_idxs = np.random.choice(range(len(val_x)), qones, replace=False, p=pred_ones/np.sum(pred_ones))
    
    rest_idxs = np.setdiff1d(range(len(val_x)), one_idxs)
    rest_idxs = np.random.choice(rest_idxs, n-one_idxs.shape[0], replace=False)
    q_idxs = np.concatenate((one_idxs, rest_idxs))
    
    train_x_new, val_x_new, train_y_new, val_y_new, valid_x, valid_y = update_datasets(train_x, val_x, train_y, val_y, q_idxs)
    

    # no weights the first round 
    Ns = train_x.shape[0]
    n_t_l = q_idxs.shape[0]
    weight_BAL = None   
    aa, sa, fa, model = retrain_model(train_x_new, train_y_new, val_x, val_y, one_idx, C)
    acc.append(aa)
    acc_one.append(sa)
    f1.append(fa)
    src_acc.append(model.model.score(train_x[one_train], train_y[one_train]))
    
    
    selected_idx = []
    
    for i in range(2, 6): 
        q_idxs = herding(model=model, train_x_new=train_x_new, val_x_new=val_x_new, train_y_new=train_y_new, budget=n)
        # q_idxs = np.argpartition(tar_infl, -n)[-n:]
        selected_idx.append(q_idxs)
        n_t_l += q_idxs.shape[0]
        weight_BAL = np.r_[np.ones(Ns), Ns/n_t_l*np.ones(n_t_l)] 
         
        train_x_new, val_x_new, train_y_new, val_y_new, valid_x, valid_y = update_datasets(train_x_new, val_x_new, train_y_new, val_y_new, q_idxs)      
        aa, sa, fa, model = retrain_model(train_x_new, train_y_new, val_x, val_y, one_idx, C, weights=weight_BAL)
        
        acc.append(aa)
        acc_one.append(sa)
        f1.append(fa)
        src_acc.append(model.model.score(train_x[one_train], train_y[one_train]))
    

    selected_labels = train_y_new[-n*5:]    
    pred = model.model.predict(val_x[one_idx])
    label = val_y[one_idx]

    sel_y_train = train_y_new[-n*5:]
    sel_x_train = train_x_new[-n*5:]

    ori_y_train = train_y_new[:-n*5]
    ori_x_train = train_x_new[:-n*5]

    sel_none = sel_y_train==0
    sel_y_train = sel_y_train[sel_none]
    sel_x_train = sel_x_train[sel_none]

    train_x_o = np.concatenate((ori_x_train, sel_x_train))
    train_y_o = np.concatenate((ori_y_train, sel_y_train))

    n_t_l = sel_y_train.shape[0]
    weight_BAL = np.r_[np.ones(Ns), Ns/n_t_l*np.ones(n_t_l)]

    # aa, sa, fa, model = retrain_model(train_x_o, train_y_o, val_x, val_y, one_idx, C, weights=weight_BAL)
    # src_acc.append(model.model.score(train_x[one_train], train_y[one_train]))
    # pred_o = model.model.predict(val_x[one_idx])
    # label_o = val_y[one_idx]

    # sel_y_train = train_y_new[-n*5:]
    # sel_x_train = train_x_new[-n*5:]

    # ori_y_train = train_y_new[:-n*5]
    # ori_x_train = train_x_new[:-n*5]

    # sel_none = sel_y_train==1
    # sel_y_train = sel_y_train[sel_none]
    # sel_x_train = sel_x_train[sel_none]

    # train_x_1 = np.concatenate((ori_x_train, sel_x_train))
    # train_y_1 = np.concatenate((ori_y_train, sel_y_train))

    # n_t_l = sel_y_train.shape[0]
    # weight_BAL = np.r_[np.ones(Ns), Ns/n_t_l*np.ones(n_t_l)]

    # aa, sa, fa, model = retrain_model(train_x_1, train_y_1, val_x, val_y, one_idx, C, weights=weight_BAL)
    # src_acc.append(model.model.score(train_x[one_train], train_y[one_train]))
    # pred_1 = model.model.predict(val_x[one_idx])
    # label_1 = val_y[one_idx]
    
    # return src_acc, acc, ori_one, acc_one, f1, ori_pred, pred, label, selected_labels, pred_o, label_o, pred_1, label_1
    return src_acc, acc, ori_one, acc_one, f1, ori_pred, pred, label, selected_labels, selected_idx

if __name__ == "__main__":

    np.random.seed(0)
    random.seed(0) 
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    cudnn.deterministic = True
    cudnn.benchmark = False


    train_x = np.load('./data_repr/'+tsk+'/source_emb.npy')
    train_y = np.load('./data_repr/'+tsk+'/source_lab.npy')
    val_x = np.load('./data_repr/'+tsk+'/target_emb.npy')
    val_y = np.load('./data_repr/'+tsk+'/target_lab.npy')
    
    
    logger.info("Load data succesful")

    ori_ones = []
    acc_ones = []
    src_accs = []
    selected_labels = []
    selected_idxs = []
    
    start = time.time()

    for j in range(126):
        src_acc, acc, ori_one, acc_one, f1, ori_pred, pred, label, lbs, idxs = active_learning(train_x, train_y, val_x, val_y, j)
        ori_ones.append(ori_one)
        acc_ones.append(acc_one)
        src_accs.append(src_acc)
        selected_labels.append(lbs)
        selected_idxs.append(idxs)

        if j == 0:
            ori_preds = ori_pred
            preds = pred
            labels = label
        else:
            ori_preds = np.concatenate((ori_preds, ori_pred))
            preds = np.concatenate((preds, pred))
            labels = np.concatenate((labels, label))

        logging.info(f"Domain adaptation for class {j} successful")
        logging.info("Original accuracy %4f - Adaptation accuracy %4f", (ori_pred == label).sum() / len(ori_pred), (pred == label).sum() / len(pred))

    np.save("/opt/kcutp/mbh_ui/active_learning/Category_Aware_DA/log/" + tsk + "/herding_selected_idxs_" + timestamp + ".npy", selected_idxs)
    logging.info("same start 1.0: %s", tsk)
    logging.info("Original accuracy %4f - Adaptation accuracy %4f", (ori_preds == labels).sum() / len(preds), (preds == labels).sum() / len(preds))
    logging.info("Total time: %4f", time.time()-start)
