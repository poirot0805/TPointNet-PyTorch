import torch
import torch.nn as nn
import torch.autograd as grad
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
import numpy as np
import time
import os.path as osp
import os
import pickle

from loader import BvhDataSet
from models.pointnet_predictor import PointNetPredictor

def get_train_stats(data_loader, use_cache=True, stats_folder=None,
                    dataset_name="train_stats"):

    stats_path = os.path.join(stats_folder, "train_stats_context.pkl")

    if use_cache and os.path.exists(stats_path):
        with open(stats_path, "rb") as fh:
            train_stats = pickle.load(fh)
        print("Train stats load from {}".format(stats_path))
    else:
        # calculate training stats
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        input_data = []
        for i, data in enumerate(data_loader, 0):
            (pose,pcd,target,data_idx) = data    # FIXME:返回需要改改

            # x = get_model_input(positions, rotations) [batch,seq,joint,dims] ->[batch,jpoint,dims]
            input_data.append(pose.cpu().numpy())

        input_data = np.concatenate(input_data, axis=0)
        print("in <get_train_stats> input shape:{}".format(input_data.shape))
        mean = np.mean(input_data, axis=0)
        std = np.std(input_data, axis=0)

        train_stats = {
            "mean": mean,
            "std": std
        }

        with open(stats_path, "wb") as fh:
            pickle.dump(train_stats, fh)

        print("Train stats wrote to {}".format(stats_path))

    return train_stats["mean"], train_stats["std"]

def get_train_stats_torch(data_loader, dtype, device,
                          use_cache=True, stats_folder=None,
                          dataset_name="train_stats"):
    mean, std = get_train_stats(data_loader, use_cache, stats_folder, dataset_name)
    mean = torch.tensor(mean, dtype=dtype, device=device)
    std = torch.tensor(std, dtype=dtype, device=device)
    return mean, std

def eval_on_dataset(model,data_loader_val,mean,std,loss_func):
    # mean,std=get_train_stats_torch(data_loader_val,dtype=data_loader_val.dataset.dtype,device="cuda:0",stats_folder=r"E:\PROJECTS\tooth\code\step-prediction\data",dataset_name="eval_stats")
    val_acc=0
    best_acc=-100
    total_loss=0
    cnt=0
    for i, data in enumerate(data_loader_val, 0):
        (pose,pcd,target, data_idx) = data # FIXME:返回类型
        
        res = evaluate(model, pose, pcd, target,mean,std)
        score=r2_score(target.detach().numpy(),res.detach().numpy())
        loss = loss_func(res, target)
        total_loss+=loss.item()
        if score>best_acc:
            best_acc=score
        val_acc+=best_acc
        cnt+=1
    print("eval---")
    print("average score:{:.4f} | best-score:{:.4f} | loss:{:.4f}".format(val_acc/cnt, best_acc,total_loss/cnt))
    print("---\n")
    return val_acc/cnt, best_acc,total_loss/cnt
        
        
def evaluate(model,pose,pcd,target,mean,std):
    with torch.no_grad():
        model.eval()
        pcd = grad.Variable(pcd).to("cuda:0")
        target = grad.Variable(target).to("cuda:0")
        # zscore
        x_zscore = (pose - mean) / std
        # calculate model output y
        res, _ = model(x_zscore, pcd)

        return res
    
def main():
	
	device="cuda:0"
	num_points = 2000
	dims = 3
	batch_size = 32
	num_epochs = 300
	lr = 0.0625
	printout = 10
	reg_weight = 0.001
	dataset_root_path = r"E:\PROJECTS\tooth\code\motion-inbetween\datasets\teeth10k"
	dataset_train_path=os.path.join(dataset_root_path,"train")
	dataset_val_path=os.path.join(dataset_root_path,"val")
	snapshot = 10
	snapshot_dir = 'snapshots'

	try:
		os.mkdir(snapshot_dir)
	except:
		pass


	# Instantiate a dataset loader
	dataset = BvhDataSet(dataset_train_path,complete_flag=False)
	data_loader = DataLoader(dataset, batch_size=batch_size,shuffle=True)
	dataset_val=BvhDataSet(dataset_val_path,complete_flag=False)
	data_loader_val = DataLoader(dataset_val, batch_size=batch_size,shuffle=False)
	print("{} clips in dataset.".format(len(dataset)))

	mean,std=get_train_stats_torch(data_loader,dtype=dataset.dtype,device="cuda:0",stats_folder=r"E:\PROJECTS\tooth\code\step-prediction\data")
	# Instantiate the network
	predictor = PointNetPredictor(num_points, dims).to(device)
	loss = nn.MSELoss()
	regularization = nn.MSELoss()
	optimizer = torch.optim.Adam(predictor.parameters(), lr=lr)
	scheduler = torch.optim.lr_scheduler.StepLR(
		optimizer, step_size=20, gamma=0.5)

	# Identity matrix for enforcing orthogonality of second transform
	identity = grad.Variable(torch.eye(64).double().cuda(), 
		requires_grad=False)

	# Some timers and a counter
	forward_time = 0.
	backprop_time = 0.
	network_time = 0.
	batch_counter = 0

	# Whether to save a snapshot
	save = False
	best_acc=-100
	best_ep=0
	print('Starting training...\n')

	# Run through all epochs
	for ep in range(num_epochs):

		if ep % snapshot == 0 and ep != 0:
			save = True

		# Update the optimizer according to the learning rate schedule
		
		for i, sample in enumerate(data_loader,0):
			(pose,pcd,target,idx)=sample
			print("pose shape:{} | pcd shape:{} | target shape:{}".format(pose.shape,pcd.shape,target.shape))

			pcd = grad.Variable(pcd).cuda()
			target = grad.Variable(target).cuda()
			pose_zscore=(pose-mean)/std
			# Record starting time
			start_time = time.time()

			# Zero out the gradients
			optimizer.zero_grad()
			predictor.train()
			# Forward pass
			pred, T2 = predictor(pose_zscore,pcd)

			# Compute forward pass time
			forward_finish = time.time()
			forward_time += forward_finish - start_time

			# Compute cross entropy loss
			pred_error = loss(pred, target)

			# Also enforce orthogonality in the embedded transform :T2(B,28,64,64) ->B(64,64)
			reg_error = regularization(
				torch.bmm(T2, T2.permute(0,1,3,2)), 
				identity.expand(*T2.shape[0:2], -1, -1))

			# Total error is the weighted sum of the prediction error and the 
			# regularization error
			total_error = pred_error + reg_weight * reg_error

			# Backpropagate
			total_error.backward()

			# Update the weights
			optimizer.step()
			scheduler.step()
			# Compute backprop time
			backprop_finish = time.time()
			backprop_time += backprop_finish - forward_finish

			# Compute network time
			network_finish = time.time()
			network_time += network_finish - start_time

			# Increment batch counter
			batch_counter += 1

			#------------------------------------------------------------------
			# Print feedback
			#------------------------------------------------------------------

			if (i+1) % printout == 0:
				# Print progress
				print('Epoch {}/{}'.format(ep+1, num_epochs))
				print('Batches {}-{}/{} (BS = {})'.format(i-printout+1, i,
					len(dataset) / batch_size, batch_size))
				print('PointClouds Seen: {}'.format(
					ep * len(dataset) + (i+1) * batch_size))
				
				# Print network speed
				print('{:16}[ {:12}{:12} ]'.format('Total Time', 'Forward', 'Backprop'))
				print('  {:<14.3f}[   {:<10.3f}  {:<10.3f} ]' \
					.format(network_time, forward_time, backprop_time))

				# Print current error
				print('{:16}[ {:12}{:12} ]'.format('Total Error', 
					'Pred Error', 'Reg Error'))
				print('  {:<14.4f}[   {:<10.4f}  {:<10.4f} ]'.format(
					total_error.data[0], pred_error.data[0], reg_error.data[0]))
				print('\n')

				# Reset timers
				forward_time = 0.
				backprop_time = 0.
				network_time = 0.

			avg_score,_,avg_loss=eval_on_dataset(predictor,data_loader_val,mean,std,loss)
			if avg_score>best_acc:
				best_acc=avg_score
				best_ep=ep
				print('Saving model snapshot...')
				save_model(predictor, snapshot_dir, ep)
				
	print("best epoch:{} best acc:{:.4f}".format(best_ep,best_acc))
 
def save_model(model, snapshot_dir, ep):
	save_path = osp.join(snapshot_dir, 'snapshot{}.params' \
		.format(ep))
	torch.save(model.state_dict(), save_path)	



if __name__ == '__main__':
	main()
