import torch
import torch.nn.functional as F
from cam.basecam import BaseCAM
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import sys

class ClusterScoreCAM2(BaseCAM):
    """
    Score-CAM with clustering: sử dụng đại diện cụm (centroid) để mask input.
    
    Args:
        model_dict: dict giống BaseCAM
        num_clusters: số lượng cụm K
    """
    def __init__(self, model_dict, num_clusters=10):
        super().__init__(model_dict)
        self.K = num_clusters

    def forward(self, input, class_idx=None, retain_graph=False):
       
        b, c, h, w = input.size()
        
        # 1) forward gốc + chọn class
        logits = self.model_arch(input)
        if class_idx is None:    
            class_idx = logits.argmax(dim=1).item()
        score = logits[0, class_idx]
        
        
        # backward để lấy activations
        self.model_arch.zero_grad()
        score.backward(retain_graph=retain_graph)
        activations = self.activations['value']  # (1, c, u, v)
        _, nc, u, v = activations.shape

        # 2) upsample + normalize từng map
        maps = []
        for i in range(nc):
            m = activations[0, i:i+1]  # (1, u, v)
            m = F.interpolate(
                m.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False
            )[0, 0]
            if m.max() == m.min():
                maps.append(torch.zeros_like(m))
            else:
                maps.append((m - m.min()) / (m.max() - m.min()))

        # stack và flatten
        all_maps = torch.stack(maps, dim=0).view(nc, -1).detach().cpu().numpy()

        # 3) clustering
        kmeans = KMeans(n_clusters=self.K, random_state=0).fit(all_maps)
        # lấy centroid và reshape về (h, w)
        rep_maps = torch.from_numpy(
            kmeans.cluster_centers_.reshape(self.K, h, w)
        ).to(activations.device)

        # 4) tính score cho từng cụm
        # cluster_scores = torch.zeros(self.K, device=activations.device)
        diffs = torch.zeros(self.K, device=activations.device)
        with torch.no_grad():
            for c_idx in range(self.K):
       
                mask = rep_maps[c_idx].unsqueeze(0).unsqueeze(0)  # (1,1,h,w)
                if mask.max() > 0:
                    mask = (mask - mask.min()) / (mask.max() - mask.min())

     
                out_mask = self.model_arch(input * mask) # (1, num_classes)
                raw_score = out_mask[0, class_idx]  
                diffs[c_idx] = raw_score - score 
                
        print("Diffs:", diffs)
        zero_percent = 0.5
        num_zero = int(self.K * zero_percent)
        if num_zero > 0:
            _, sorted_idx = torch.sort(diffs, descending=False)
        lowest_idx = sorted_idx[:num_zero]
        # gán -inf để softmax → 0 
        diffs[lowest_idx] = float('-inf')

        # 4.2) Softmax trên diffs để chuẩn hoá trọng số
        cluster_scores = F.softmax(diffs, dim=0)
        
        print("Cluster scores after softmax:", cluster_scores)
                
                
                # masked_input = input * mask  
                
                # img_np = (
                # masked_input
                # .squeeze(0)               # (C, H, W)
                # .permute(1, 2, 0)         # (H, W, C)
                # .cpu()                    # lên CPU
                # .detach()
                # .numpy()
                # )
            
                # img_norm = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-6)

                # 3) vẽ
                # plt.imsave(
                #     f"/home/infres/xnguyen-24/XAI/ScoreCAM_cluster/pics/masked_input_cluster{c_idx}_mpl.png",
                #     img_norm
                # )
                        
                
                # raw_score = out_mask[0, class_idx]               # logit của lớp mục tiêu
                # diff = raw_score - score             # chênh lệch so với gốc

                # nếu diff > 0 thì softmax và lấy xác suất, ngược lại giữ 0
                # probs = F.softmax(diff, dim=1)
                # cluster_scores[c_idx] = diff
                # if diff.item() > 0:
                #     probs = F.softmax(out_mask, dim=1)
                #     cluster_scores[c_idx] = probs[0, class_idx]
                # else:
                #     cluster_scores[c_idx] = 0
        
        # if torch.all(cluster_scores == 0):
        #     print("Warning: all cluster scores are zero.")
        #     sys.exit(1)
        # else:
        #     num_zeros = int((cluster_scores == 0).sum().item())
        #     print(f"Number of zero cluster scores: {num_zeros}")
        
        
        # print(f"Cluster scores: {cluster_scores}")
        
        # cluster_scores = F.softmax(cluster_scores, dim=0)
        
        # print("Softmax cluster scores:", cluster_scores)
        # 5) reconstruct saliency_map: dùng rep * score
        saliency_map = torch.zeros(1, 1, h, w, device=activations.device)
        for lbl in range(self.K): 
            saliency_map += cluster_scores[lbl] * rep_maps[lbl].unsqueeze(0).unsqueeze(0)

        # 6) post-process
        saliency_map = F.relu(saliency_map)
        mn, mx = saliency_map.min(), saliency_map.max()
        if mn == mx:
            return None
        saliency_map = (saliency_map - mn) / (mx - mn)
        
        return saliency_map

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)
