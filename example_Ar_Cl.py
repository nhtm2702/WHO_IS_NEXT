import numpy as np
from model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.ensemble import GradientBoostingRegressor
import time
import logging
import random
import torch
import torch.backends.cudnn as cudnn
import datetime

import warnings
warnings.filterwarnings("ignore")

def train_and_eval(train_x, train_y, val_x, val_y, one_idx, C):
    model = LogisticRegression(C)
    model.fit(train_x, train_y)
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
    
    #weights[:-n] = 1 / weights[:-n].shape[0]
    #weights[-n:] = 1 / n
    model.fit(train_x_new, train_y_new, sample_weight=weights)
    
    return model.model.score(val_x, val_y), model.model.score(val_x[one_idx], val_y[one_idx]), f1_score(model.model.predict(val_x), val_y, average='binary'), model

def infl_pred(model, train_x_new, val_x_new, train_y_new, val_y_new, valid_x, valid_y, weights=None):
    # ori_util_loss_val = model.log_loss(valid_x, valid_y)
    # pred_train, _ = model.pred(train_x_new)

    train_total_grad, train_indiv_grad = model.grad(train_x_new, train_y_new, sample_weight=weights)
    util_loss_total_grad, acc_loss_indiv_grad = model.grad(valid_x, valid_y)

    hess = model.hess(train_x_new, sample_weight=weights)

    util_grad_hvp = model.get_inv_hvp(hess, util_loss_total_grad)
    util_pred_infl = train_indiv_grad.dot(util_grad_hvp)

    # print(util_pred_infl.shape)
    # compute source infl and predict target infl
    src_infl = util_pred_infl
    src_infl = list(src_infl.reshape(-1))
    reg = GradientBoostingRegressor(n_estimators=1000, 
                                    max_depth=6, 
                                    learning_rate=0.05, 
                                    max_features=0.1, 
                                    min_samples_split=25, 
                                    min_samples_leaf=25).fit(train_x_new, src_infl)

    tar_infl = reg.predict(val_x_new)
    pred_infl = reg.predict(train_x_new)
    

    return src_infl, pred_infl, tar_infl

def active_learning(train_x, train_y, val_x, val_y, one):

    #enhance the performance for a specific category
    train_y = np.where(train_y == one, 1, 0)
    val_y = np.where(val_y == one, 1, 0)
    one_train = train_y==1
    one_idx = val_y==1
    
    L2_WEIGHT = 1e-4
    C = 1 / (train_x[0].shape[0] * L2_WEIGHT)
    
    src_acc = []
    acc, acc_one, f1 = [], [], []
    aa, sa, fa, model = train_and_eval(train_x, train_y, val_x, val_y, one_idx, C)
    ori_pred = model.model.predict(val_x[one_idx])
    ori_one = sa
    
    src_acc.append(model.model.score(train_x[one_train], train_y[one_train]))
    
    n = int(val_x.shape[0]*0.01)

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
        src_infl, pred_infl, tar_infl = infl_pred(model, train_x_new, val_x_new, train_y_new, val_y_new, valid_x, valid_y, weights=weight_BAL)
        q_idxs = np.argpartition(tar_infl, -n)[-n:]
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

    aa, sa, fa, model = retrain_model(train_x_o, train_y_o, val_x, val_y, one_idx, C, weights=weight_BAL)
    src_acc.append(model.model.score(train_x[one_train], train_y[one_train]))
    pred_o = model.model.predict(val_x[one_idx])
    label_o = val_y[one_idx]

    sel_y_train = train_y_new[-n*5:]
    sel_x_train = train_x_new[-n*5:]

    ori_y_train = train_y_new[:-n*5]
    ori_x_train = train_x_new[:-n*5]

    sel_none = sel_y_train==1
    sel_y_train = sel_y_train[sel_none]
    sel_x_train = sel_x_train[sel_none]

    train_x_1 = np.concatenate((ori_x_train, sel_x_train))
    train_y_1 = np.concatenate((ori_y_train, sel_y_train))

    n_t_l = sel_y_train.shape[0]
    weight_BAL = np.r_[np.ones(Ns), Ns/n_t_l*np.ones(n_t_l)]

    aa, sa, fa, model = retrain_model(train_x_1, train_y_1, val_x, val_y, one_idx, C, weights=weight_BAL)
    src_acc.append(model.model.score(train_x[one_train], train_y[one_train]))
    pred_1 = model.model.predict(val_x[one_idx])
    label_1 = val_y[one_idx]
    
    # return src_acc, acc, ori_one, acc_one, f1, ori_pred, pred, label, selected_labels, pred_o, label_o, pred_1, label_1
    return src_acc, acc, ori_one, acc_one, f1, ori_pred, pred, label, selected_labels, selected_idx

if __name__ == "__main__":
    
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler("/opt/kcutp/mbh_ui/active_learning/Category_Aware_DA/log/example_" + timestamp + ".log",  mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info("Session started")
    
    tsk = 'Ar2Cl' # 'Ar2Cl'
    np.random.seed(0)
    # random.seed(0)
    # torch.manual_seed(0)
    # cudnn.deterministic = True


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

    for j in range(65):
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
        
    np.save("/opt/kcutp/mbh_ui/active_learning/Category_Aware_DA/log/selected_idxs_" + timestamp + ".npy", selected_idxs)
    logging.info("same start 1.0: %s", tsk)
    logging.info("Orginial accuracy %4f - Adaptation accuracy %4f", (ori_preds == labels).sum() / len(preds), (preds == labels).sum() / len(preds))
    logging.info("Total time: %4f", time.time()-start)