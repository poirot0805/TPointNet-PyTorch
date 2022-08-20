import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as grad

from models.pointnet_base import PointNetBase


class PointNetPredictor(nn.Module):

    def __init__(self, num_points=2000, K=3):
		# Call the super constructor
        super(PointNetPredictor, self).__init__()

		# Local and global feature extractor for PointNet
        self.base = PointNetBase(num_points, K)

		# Classifier for ShapeNet
        self.predictor = nn.Sequential(
			nn.Linear(1038, 512),
			nn.BatchNorm1d(512),
			nn.ReLU(),
			nn.Dropout(0.1),
			nn.Linear(512, 256),
			nn.BatchNorm1d(256),
			nn.ReLU(),
			nn.Dropout(0.1),
			nn.Linear(256, 1))

    def forward(self,x,pcd):
		# Output should be B x 1024
        # pcd:[B,28,3,2000] - >[B,K,N]
        B=pcd.shape[0]
        pcd_feature=torch.zeros([B,28,1024],dtype=pcd.dtype,device=pcd.device)
        t2_feature=torch.zero([B,28,64,64],dtype=pcd.dtype,device=pcd.device)
        for i in range(28):
            gfeatures, _, T2 = self.base(pcd[:,i:i+1].squeeze())    # (B,1024)
            pcd_feature[:,i:i+1]=gfeatures.unsqueeze(1)
            t2_feature[:,i:i+1]=T2.unsqueeze(1)
        
        # x:[B,28,14]
        y=torch.cat([pcd_feature,x],dim=-1)
        
        ans=self.predictor(y)
        res = nn.AvgPool2d([28,1])
        return res,t2_feature
		# Returns a B x 40 
