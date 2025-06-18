import torch
import torch.nn.functional as F
from cam.basecam import BaseCAM
from sklearn.cluster import KMeans

class ClusterScoreCAM(BaseCAM):
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
        cluster_scores = torch.zeros(self.K, device=activations.device)
        with torch.no_grad():
            for c_idx in range(self.K):
                mask = rep_maps[c_idx].unsqueeze(0).unsqueeze(0)  # (1,1,h,w)
                # normalize mask
                if mask.max() > 0:
                    mask = (mask - mask.min()) / (mask.max() - mask.min())
                out = self.model_arch(input * mask)
                probs = F.softmax(out, dim=1)
                cluster_scores[c_idx] = probs[0, class_idx]

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
