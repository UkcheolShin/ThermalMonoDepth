import argparse
import cv2
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import os
from tqdm import tqdm
from path import Path

parser = argparse.ArgumentParser(description="Depth options")
parser.add_argument("--dataset", required=True, help="VIVID", choices=['VIVID'], type=str)
parser.add_argument("--scene", required=True, help="scene type for VIVID", choices=['indoor', 'outdoor'], type=str)
parser.add_argument("--pred_depth", required=True, help="depth predictions npy", type=str)
parser.add_argument("--gt_depth", required=True, help="gt depth folder for VIVID", type=str)
parser.add_argument("--ratio_name", help="names for saving ratios", type=str)


args = parser.parse_args()

def compute_depth_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    Args:
        gt (N): ground truth depth
        pred (N): predicted depth
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    log10 = np.mean(np.abs((np.log10(gt) - np.log10(pred))))

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

class DepthEval():
    def __init__(self):

        self.min_depth = 1e-3

        if args.dataset == 'VIVID':
            if args.scene == 'indoor' :
                self.max_depth = 10.
            elif args.scene == 'outdoor' :
                self.max_depth = 80.

    def main(self):
        pred_depths = []

        """ Get result """
        # Read precomputed result
        pred_depths = np.load(os.path.join(args.pred_depth))

        """ Evaluation """
        if args.dataset == 'VIVID':
            gt_depths = []
            for gt_f in sorted(Path(args.gt_depth).files("*.npy")):
                gt_depths.append(np.load(gt_f))

        pred_depths = self.evaluate_depth(gt_depths, pred_depths, eval_mono=True)


    def evaluate_depth(self, gt_depths, pred_depths, eval_mono=True):
        """evaluate depth result
        Args:
            gt_depths (NxHxW): gt depths
            pred_depths (NxHxW): predicted depths
            split (str): data split for evaluation
                - depth_eigen
            eval_mono (bool): use median scaling if True
        """
        errors = []
        ratios = []
        resized_pred_depths = []

        print("==> Evaluating depth result...")
        for i in tqdm(range(pred_depths.shape[0])):
            if pred_depths[i].mean() != -1:
                gt_depth = gt_depths[i]
                gt_height, gt_width = gt_depth.shape[:2]

                # resizing prediction (based on inverse depth)
                pred_inv_depth = 1 / (pred_depths[i] + 1e-6)
                pred_inv_depth = cv2.resize(pred_inv_depth, (gt_width, gt_height))
                pred_depth = 1 / (pred_inv_depth + 1e-6)

                mask = np.logical_and(gt_depth > self.min_depth, gt_depth < self.max_depth)
                val_pred_depth = pred_depth[mask]
                val_gt_depth = gt_depth[mask]

                # median scaling is used for monocular evaluation
                ratio = 1
                if eval_mono:
                    ratio = np.median(val_gt_depth) / np.median(val_pred_depth)
                    ratios.append(ratio)
                    val_pred_depth *= ratio

                resized_pred_depths.append(pred_depth * ratio)

                val_pred_depth[val_pred_depth < self.min_depth] = self.min_depth
                val_pred_depth[val_pred_depth > self.max_depth] = self.max_depth

                errors.append(compute_depth_errors(val_gt_depth, val_pred_depth))

        if eval_mono:
            ratios = np.array(ratios)
            med = np.median(ratios)
            print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))
            print(" Scaling ratios | mean: {:0.3f} +- std: {:0.3f}".format(np.mean(ratios), np.std(ratios)))
            if args.ratio_name:
                np.savetxt(args.ratio_name, ratios, fmt='%.4f')

        mean_errors = np.array(errors).mean(0)
        print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")

        return resized_pred_depths


eval = DepthEval()
eval.main()
