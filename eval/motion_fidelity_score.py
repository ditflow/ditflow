# Adapted from https://github.com/diffusion-motion-transfer/diffusion-motion-transfer
import argparse
from pathlib import Path
import numpy as np
import torch
from cotracker.utils.visualizer import read_video_from_path
from einops import rearrange
import imageio


def get_similarity_matrix(tracklets1, tracklets2):
    displacements1 = tracklets1[:, 1:] - tracklets1[:, :-1]
    displacements1 = displacements1 / displacements1.norm(dim=-1, keepdim=True)

    displacements2 = tracklets2[:, 1:] - tracklets2[:, :-1]
    displacements2 = displacements2 / displacements2.norm(dim=-1, keepdim=True)

    similarity_matrix = torch.einsum("ntc, mtc -> nmt", displacements1, displacements2).mean(dim=-1)
    return similarity_matrix


def get_score(similarity_matrix):
    similarity_matrix_eye = similarity_matrix - torch.eye(similarity_matrix.shape[0]).to(similarity_matrix.device)
    # for each row find the most similar element
    max_similarity, _ = similarity_matrix_eye.max(dim=1)
    average_score = max_similarity.mean()
    return {
        "average_score": average_score.item(),
    }


def get_tracklets(model, video, mask=None):
    video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float().cuda()
    pred_tracks_small, pred_visibility_small = model(video, grid_size=55, segm_mask=mask)
    pred_tracks_small = rearrange(pred_tracks_small, "b t l c -> (b l) t c ")
    return pred_tracks_small


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--original_video_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="./results")
    opt = parser.parse_args()
    
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
    model = model.cuda()

    box_mask = None
    original_video = read_video_from_path(opt.original_video_path)
    original_tracklets = get_tracklets(model, original_video, mask=box_mask)
    
    video = np.array(imageio.mimread(opt.video_path))
    edit_tracklets = get_tracklets(model, video, mask=box_mask)
    
    min_t = min(edit_tracklets.shape[1], original_tracklets.shape[1])
    edit_tracklets = edit_tracklets[:, :min_t]
    original_tracklets = original_tracklets[:, :min_t]
    # print(edit_tracklets.shape, original_tracklets.shape)
    similarity_matrix = get_similarity_matrix(edit_tracklets, original_tracklets)
    similarity_scores_dict = get_score(similarity_matrix)

    results = similarity_scores_dict["average_score"]
    print("Tracklets score: ", results)
    Path(opt.output_path).mkdir(parents=True, exist_ok=True)
    torch.save(results, f"{opt.output_path}/tracklets.pt")
