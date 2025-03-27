import torch
import torch.nn.functional as F
from einops import rearrange

def compute_motion_flow(q, k, h=30, w=45, temp=5, nframes=6, argmax=False):
    """Compute Attention Motion Flow (AMF)"""
    def compute_displacement(A):
        device = A.device
        
        if argmax:
            matches = A.argmax(dim=-1)
        
            def to_coordinates(indices, width=w):
                x = indices % width
                y = indices // width
                return x, y

            x1, y1 = to_coordinates(torch.arange(A.shape[0], device=device))
            x2, y2 = to_coordinates(matches)
            dx = x2 - x1
            dy = y2 - y1
            displacements = torch.stack((dx, dy), dim=-1)

        else:
            # Create grid of relative coordinates
            y_coords, x_coords = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
            relative_x = x_coords.flatten().unsqueeze(0) - x_coords.flatten().unsqueeze(1)
            relative_y = y_coords.flatten().unsqueeze(0) - y_coords.flatten().unsqueeze(1)
            
            # Compute weighted average
            displacement_x = (relative_x * A).sum(dim=1)
            displacement_y = (relative_y * A).sum(dim=1)
            
            displacements = torch.stack([displacement_x, displacement_y], dim=-1)
        return displacements

    A_all = torch.matmul(q[-1,:,226:,:], k[-1,:,226:,:].transpose(-1,-2)) / torch.sqrt(torch.tensor(q.shape[-1], dtype=q.dtype, device=q.device))
    A_all = A_all.mean(0, keepdim=True)
    
    total_predicted_flows = 0
    for head in range(A_all.shape[0]):
        predicted_flows = []
        A_head = A_all[head]

        # Softmax per frame
        A_head = rearrange(A_head, 's (f hw) -> s f hw', f=nframes)
        A_head = F.softmax(A_head*temp, dim=-1)
        A_head = rearrange(A_head, '(f1 s1) f2 s2 -> f1 f2 s1 s2', f1=nframes, f2=nframes, s1=h*w, s2=h*w)

        for frame_i in range(nframes):
            for frame_j in range(nframes):
                displacement = compute_displacement(A_head[frame_i, frame_j])
                predicted_flows.append(displacement)

        predicted_flows = torch.stack(predicted_flows, dim=0)
        total_predicted_flows += predicted_flows

    return total_predicted_flows / A_all.shape[0]