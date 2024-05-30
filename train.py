import argparse
import random
import numpy as np
import torch
import os
import importlib

import utils
from dataset import TrajectoryDataset
from torch.utils.data import DataLoader
from utils import *


from model import TrajectoryModel
from torch import optim
import torch.nn.functional as F

from utils import get_motion_modes

import matplotlib.pyplot as plt
import time
import logging
from datetime import datetime



parser = argparse.ArgumentParser() 

parser.add_argument('--dataset_path', type=str, default='./dataset/')
#parser.add_argument('--log_path',type=str, default='./log/')
parser.add_argument('--dataset_name', type=str, default='zara2')
parser.add_argument("--hp_config", type=str, default='config/zara2.py', help='hyper-parameter')
parser.add_argument('--lr_scaling', action='store_true', default=False)
parser.add_argument('--num_works', type=int, default=8)
parser.add_argument('--obs_len', type=int, default=8)
parser.add_argument('--pred_len', type=int, default=12)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--data_scaling', type=list, default=[1.9, 0.4])
parser.add_argument('--dist_threshold', type=float, default=2)
parser.add_argument('--checkpoint', type=str, default='./checkpoint/')

args = parser.parse_args()
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
print(args)

# python train.py --dataset_name sdd --gpu 0 --hp_config config/sdd.py
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

spec = importlib.util.spec_from_file_location("hp_config", args.hp_config)
hp_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hp_config)


train_dataset = TrajectoryDataset(dataset_path=args.dataset_path, dataset_name=args.dataset_name,
                                  dataset_type='train', translation=True, rotation=True,
                                  scaling=True, obs_len=args.obs_len, 
                                  dist_threshold=hp_config.dist_threshold, smooth=False) 

test_dataset = TrajectoryDataset(dataset_path=args.dataset_path, dataset_name=args.dataset_name,
                                  dataset_type='test', translation=True, rotation=True,
                                  scaling=False, obs_len=args.obs_len)

motion_modes_file = args.dataset_path + args.dataset_name + '_motion_modes.pkl'

if not os.path.exists(motion_modes_file):
    print('motionm modes generating ... ')
    motion_modes, sub_modes = get_motion_modes(train_dataset, args.obs_len, args.pred_len, hp_config.n_clusters, args.dataset_path, args.dataset_name,
                                    smooth_size=hp_config.smooth_size, random_rotation=hp_config.random_rotation, traj_seg=hp_config.traj_seg)
    motion_modes = torch.tensor(motion_modes, dtype=torch.float32).cuda()
    sub_modes = torch.tensor(sub_modes, dtype=torch.float32).cuda()

    
train_loader = DataLoader(train_dataset, collate_fn=train_dataset.coll_fn, batch_size=hp_config.batch_size, shuffle=True, num_workers=args.num_works)
test_loader = DataLoader(test_dataset, collate_fn=test_dataset.coll_fn, batch_size=hp_config.batch_size, shuffle=True, num_workers=args.num_works)

if os.path.exists(motion_modes_file):
    print('motion modes loading ... ')
    import pickle
    f = open(args.dataset_path + args.dataset_name + '_motion_modes.pkl', 'rb+')
    motion_modes = pickle.load(f)
    f.close()
    motion_modes = torch.tensor(motion_modes, dtype=torch.float32).cuda()
    f1 = open('./sub_motion_modes/eth_submotion_modes.pkl','rb+')
    sub_modes = pickle.load(f1)
    f1.close()
    sub_modes = torch.tensor(sub_modes, dtype=torch.float32).cuda()


model = TrajectoryModel(in_size=2, obs_len=args.obs_len, pred_len=args.pred_len, embed_size=hp_config.model_hidden_dim, 
enc_num_layers=2, int_num_layers_list=[1,1], heads=4, forward_expansion=2)
model = model.cuda()

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=hp_config.lr)
reg_criterion = torch.nn.SmoothL1Loss().cuda()
cls_criterion = torch.nn.CrossEntropyLoss().cuda()

if args.lr_scaling:
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500, 600], gamma=0.5)# milestones=[270, 400], gamma=0.5


def get_cls_label(gt, motion_modes,sub_modes, soft_label=True):

    # motion_modes [K pred_len 2]
    # gt [B pred_len 2]

    gt = gt.reshape(gt.shape[0], -1).unsqueeze(1)  # [B 1 pred_len*2]
    motion_modes = motion_modes.reshape(motion_modes.shape[0], -1).unsqueeze(0)  # [1 K pred_len*2]
    distance = torch.norm(gt - motion_modes, dim=-1)  # [B K]
    soft_label = F.softmax(-distance, dim=-1) # [B K]
    closest_mode_indices = torch.argmin(distance, dim=-1) # [B]

    sub_modes = sub_modes.reshape(sub_modes.shape[0], -1).unsqueeze(0)
    idx_sub = (closest_mode_indices == 39).nonzero()
    gt_sub = gt[idx_sub[:, 0]]
    distance1 = torch.norm(gt_sub - sub_modes, dim=-1)
    sub_closest_mode_indices = torch.argmin(distance1, dim=-1)


    return soft_label, closest_mode_indices, sub_closest_mode_indices


def train( epoch, model, cls_criterion,optimizer, train_dataloader, motion_modes,sub_modes):
    model.train()
    total_loss = []
    for i, (ped, neis, mask) in enumerate(train_dataloader):
        ped = ped.cuda()
        neis = neis.cuda()
        mask = mask.cuda() 

        if args.dataset_name == 'eth':
            ped[:, :, 0] = ped[:, :, 0] * args.data_scaling[0]
            ped[:, :, 1] = ped[:, :, 1] * args.data_scaling[1]
        
        scale = torch.randn(ped.shape[0])*0.05+1
        scale = scale.cuda()
        scale = scale.reshape(ped.shape[0], 1, 1)
        ped = ped * scale
        scale = scale.reshape(ped.shape[0], 1, 1, 1)
        neis = neis * scale

        ped_obs = ped[:, :args.obs_len]
        gt = ped[:, args.obs_len:]
        neis_obs = neis[:, :, :args.obs_len]

        with torch.no_grad():
            soft_label, closest_mode_indices,sub_closest_mode_indices= get_cls_label(gt, motion_modes,sub_modes)
        
        optimizer.zero_grad()

        """
        velocity:
        """
        pred_traj_vel,pred_traj_ref_vel, scores,int_feat = model(ped_obs, neis_obs, motion_modes, mask, closest_mode_indices)#,ped_obs_feat

        pred_traj_vel = pred_traj_vel.reshape(pred_traj_vel.shape[0],12,2)
        pred_traj = [ped_obs[:,-1,:] + pred_traj_vel[:,0]]
        for i in range (1,12):
            next_coor = pred_traj[-1] + pred_traj_vel[:,i]
            pred_traj.append(next_coor)
        pred_traj = torch.stack(pred_traj, dim=1)

        pred_traj_ref_vel = pred_traj_ref_vel.reshape(pred_traj_ref_vel.shape[0], 12, 2)
        pred_traj_ref = [ped_obs[:, -1, :] + pred_traj_ref_vel[:, 0]]
        for i in range(1, 12):
            next_coor1 = pred_traj_ref[-1] + pred_traj_ref_vel[:, i]
            pred_traj_ref.append(next_coor1)
        pred_traj_ref = torch.stack(pred_traj_ref, dim=1)


        pred_traj = pred_traj.reshape(pred_traj.shape[0], 12, 2)
        reg_loss1 = utils.ewta_loss(pred_traj, gt)

        pred_traj_ref = pred_traj_ref.reshape(pred_traj_ref.shape[0], 12, 2)
        reg_loss2 = utils.ewta_loss(pred_traj_ref, gt)

        clf_loss = cls_criterion(scores.squeeze(), soft_label)

        contrastive_loss = utils.sub_contrastive_loss(int_feat, closest_mode_indices,sub_closest_mode_indices)

        pre_loss = reg_loss1 +  clf_loss + reg_loss2
        loss = 1 * pre_loss + 1 * contrastive_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss.append(loss.item())

    return total_loss, reg_loss1 ,reg_loss2, clf_loss, contrastive_loss


def vis_predicted_trajectories(obs_traj, gt, pred_trajs, pred_probabilities, min_index):


    # obs_traj [B T_obs 2]
    # gt [B T_pred 2]
    # pred_trajs [B 20 T_pred 2]
    # pred_probabilities [B 20]

    for i in range(obs_traj.shape[0]):
        plt.clf()
        curr_obs = obs_traj[i].cpu().numpy() # [T_obs 2]
        curr_gt = gt[i].cpu().numpy()
        curr_preds = pred_trajs[i].cpu().numpy()
     
        curr_pros = pred_probabilities[i].cpu().numpy()
        curr_min_index = min_index[i].cpu().numpy()
        obs_x = curr_obs[:, 0]
        obs_y = curr_obs[:, 1]
        gt_x = np.concatenate((obs_x[-1:], curr_gt[:, 0]))
        gt_y = np.concatenate((obs_y[-1:], curr_gt[:, 1]))
        plt.plot(obs_x, obs_y, marker='o', color='green')
        plt.plot(gt_x, gt_y, marker='o', color='blue')
        plt.scatter(gt_x[-1], gt_y[-1], marker='*', color='blue', s=300)
       
        for j in range(curr_preds.shape[0]):
        
            pred_x = np.concatenate((obs_x[-1:], curr_preds[j][:, 0]))
            pred_y = np.concatenate((obs_y[-1:], curr_preds[j][:, 1]))
            if j == curr_min_index:
                plt.plot(pred_x, pred_y, ls='-.', lw=2.0, color='red')
                plt.scatter(pred_x[-1], pred_y[-1], marker='*', color='orange', s=300)
            else:
                plt.plot(pred_x, pred_y, ls='-.', lw=0.5, color='red')
                plt.scatter(pred_x[-1], pred_y[-1], marker='*', color='red', s=300)
            plt.text(pred_x[-1], pred_y[-1], str("%.2f" % curr_pros[j]),  ha='center')
            
        
        plt.tight_layout()
        save_path = './fig/' + args.dataset_name
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(save_path + '/' + str(time.time()) + '.png')
        

    return


def test(model, test_dataloader, motion_modes):

    model.eval()

    ADE = 0
    FDE = 0
    num_traj = 0
    

    for (ped, neis, mask) in test_dataloader:

        ped = ped.cuda()
        neis = neis.cuda()
        mask = mask.cuda() 

        ped_obs = ped[:, :args.obs_len]
        gt = ped[:, args.obs_len:]
        neis_obs = neis[:, :, :args.obs_len]

        with torch.no_grad():
            
            num_traj += ped_obs.shape[0]


            # start = time.perf_counter()

            pred_trajs_vel, scores = model(ped_obs, neis_obs, motion_modes, mask, None, test=True)
            # end = time.perf_counter()
            # print('infer_time', end - start)
            # top_k_scores = torch.topk(scores, k=20, dim=-1).values
            # top_k_scores = F.softmax(top_k_scores, dim=-1)
            pred_trajs_vel = pred_trajs_vel.reshape(pred_trajs_vel.shape[0], pred_trajs_vel.shape[1], gt.shape[1], 2) # [B pred_len*2]
            pred_trajs = []
            for j in range(pred_trajs_vel.shape[1]):
                pred_trajs1 = [ped_obs[:,-1,:] + pred_trajs_vel[:,j,0]]
                for i in range(1, 12):
                    next_coor = pred_trajs1[-1] + pred_trajs_vel[:,j,i]
                    pred_trajs1.append(next_coor)
                pred_trajs1 = torch.stack(pred_trajs1, dim=1)
                pred_trajs.append(pred_trajs1)
            pred_trajs = torch.stack(pred_trajs, dim=1)


            pred_trajs = pred_trajs.reshape(pred_trajs.shape[0], pred_trajs.shape[1], gt.shape[1], 2)
            gt_ = gt.unsqueeze(1)
            norm_ = torch.norm(pred_trajs - gt_, p=2, dim=-1)
            ADE_ = torch.mean(norm_, dim=-1)
            FDE_ = norm_[:, :, -1]
            min_ADE, min_ADE_index = torch.min(ADE_, dim=-1)
            min_FDE, min_FDE_index = torch.min(FDE_, dim=-1)
            # vis_predicted_trajectories(ped_obs, gt, pred_trajs.reshape(pred_trajs.shape[0], pred_trajs.shape[1], gt.shape[-2], -1), top_k_scores,min_fde_index)

            # b-ade/fde
            # batch_index = torch.LongTensor(range(top_k_scores.shape[0])).cuda() 
            # min_ade_p = top_k_scores[batch_index, min_ade_index]
            # min_fde_p = top_k_scores[batch_index, min_fde_index]
            # min_ade = min_ade + (1 - min_ade_p)**2
            # min_fde = min_fde + (1 - min_fde_p)**2

            min_ADE = torch.sum(min_ADE)
            min_FDE = torch.sum(min_FDE)
            ADE += min_ADE.item()
            FDE += min_FDE.item()

    ADE = ADE / num_traj
    FDE = FDE / num_traj
    return ADE, FDE, num_traj

min_ADE = 99
min_FDE = 99


# log_directory = args.log_path + args.dataset_name
# if not os.path.exists(log_directory):
#     os.makedirs(log_directory)
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# log_path = os.path.join(log_directory, f'subcon_{timestamp}.log')
# logging.basicConfig(filename=log_path, level=logging.INFO)

for ep in range(hp_config.epoch):
    num_samples = len(train_loader.dataset)
    #print("Number of samples in training DataLoader:", num_samples)

    total_loss,reg_loss1, reg_loss2, clf_loss, contrastive_loss = train( ep, model, reg_criterion, cls_criterion, optimizer, train_loader, motion_modes,sub_modes)

    ADE, FDE, num_traj = test(model, test_loader, motion_modes)

    if args.lr_scaling:
        scheduler.step()

    if not os.path.exists(args.checkpoint + args.dataset_name):
        os.makedirs(args.checkpoint + args.dataset_name)

    if min_FDE + min_ADE > ADE + FDE:
        min_FDE = FDE
        min_ADE = ADE
        min_FDE_epoch = ep
        torch.save(model.state_dict(), args.checkpoint + args.dataset_name + '/best.pth')

    train_loss = sum(total_loss) / len(total_loss)

    print('epoch:', ep, 'data_set:', args.dataset_name, 'total_loss:', train_loss)
    # print('total_loss:', train_loss, 'reg_loss: ', loss_pre.item(), 'contrastive_loss:', contrastive_loss.item())
    print('total_loss:', train_loss, 'reg_loss1: ', reg_loss1.item(),'reg_loss2: ', reg_loss2.item(), 'clf_loss:', clf_loss.item(), 'contrastive_loss:',contrastive_loss.item())  # ,'mse_loss:',mse_loss.item())#, 'contrastive_loss:',contrastive_loss.item()
    print('epoch:', ep, 'ADE:', ADE, 'FDE:', FDE, 'min_ADE:', min_ADE, 'min_FDE:', min_FDE, 'min_FDE_epoch:',min_FDE_epoch, 'num_traj:', num_traj)
    #logging.shutdown()




