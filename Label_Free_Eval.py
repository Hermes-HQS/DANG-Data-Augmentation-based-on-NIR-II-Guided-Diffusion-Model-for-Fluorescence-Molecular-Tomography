import numpy as np
from pathlib import Path

class FMTEvaluator:
    """
    Evaluator class for FMT reconstruction models
    Metrics:
        - Dice score for segmentation accuracy
        - BCE (Barycentric coordinate error)
        - CNR (Contrast-to-Noise Ratio)
    """
    def __init__(self, brain_coords_path='Tecplot_Data/BrainNodes.npy'):
        """Initialize evaluator with brain coordinates"""
        self.brain_coords = self._load_brain_coords(brain_coords_path)

    def _load_brain_coords(self, path):
        """Load brain coordinates from numpy file"""
        return np.load(path)

    def dice_score(self, pred, target, threshold, save_path=None):
        """
        Calculate Dice similarity coefficient
        Args:
            pred: Model predictions
            target: Ground truth values
            threshold: Threshold for binary mask
            save_path: Optional path to save results
        Returns:
            dice_list: Array of Dice scores
        """
        pred_mask = (pred > threshold).astype(np.float32)
        target_mask = (target > threshold).astype(np.float32)

        intersection = np.absolute(np.sum(pred_mask * target_mask, axis=-1))
        total = np.absolute(np.sum(pred_mask, axis=-1)) + np.absolute(np.sum(target_mask, axis=-1))
        dice_list = (2. * intersection + 0.001) / (total + 0.001)

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(save_path / "dice_scores.txt", dice_list)

        return dice_list

    def bce_error(self, pred, target, threshold, save_path=None):
        """
        Calculate Barycentric Coordinate Error
        Args:
            pred: Model predictions
            target: Ground truth values
            threshold: Threshold for masking
            save_path: Optional path to save results
        Returns:
            bce_errors: Array of BCE values
        """
        pred_masked = (pred > threshold).astype(np.float32) * pred
        target_masked = (target > threshold).astype(np.float32) * target

        pred_bc = (np.matmul(pred_masked, self.brain_coords)) / (
            np.sum(pred_masked**0.2, axis=-1, keepdims=True) + 2.003)
        target_bc = (np.matmul(target_masked, self.brain_coords)) / (
            np.sum(target_masked**0.2, axis=-1, keepdims=True) + 2.003)

        bce_errors = np.linalg.norm(pred_bc - target_bc, ord=2, axis=-1) * 0.3

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(save_path / "bce_errors.txt", bce_errors)

        return bce_errors

    def cnr(self, pred):
        """
        Calculate Contrast-to-Noise Ratio
        Args:
            pred: Model predictions
        Returns:
            cnr: Contrast-to-Noise Ratio value
        """
        sorted_pred = np.sort(pred.flatten())
        largest_values = sorted_pred[-10:]
        smallest_values = sorted_pred[:50]
        
        mean_largest = np.mean(largest_values)
        mean_smallest = np.mean(smallest_values)
        
        return mean_largest / (mean_smallest + 1e-2)

def evaluate_model(model, test_loader, device, config, save_dir=None):
    """
    Evaluate FMT reconstruction model using multiple metrics
    Args:
        model: Trained FMT model
        test_loader: DataLoader with test data
        device: Device to run evaluation on
        config: Configuration object containing evaluation parameters
        save_dir: Optional directory to save results
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    evaluator = FMTEvaluator(brain_coords_path=config.brain_coords_path)
    all_dice = []
    all_bce = []
    all_cnr = []

    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.cpu().numpy()
            
            outputs = model(inputs).cpu().numpy()
            
            if 'dice' in config.eval_metrics:
                dice_scores = evaluator.dice_score(
                    outputs, 
                    targets, 
                    config.eval_threshold
                )
                all_dice.extend(dice_scores)
            
            if 'bce' in config.eval_metrics:
                bce_errors = evaluator.bce_error(
                    outputs, 
                    targets, 
                    config.eval_threshold
                )
                all_bce.extend(bce_errors)
            
            if 'cnr' in config.eval_metrics:
                cnr_values = [evaluator.cnr(output) for output in outputs]
                all_cnr.extend(cnr_values)

    results = {}
    if all_dice:
        results['dice'] = {'mean': np.mean(all_dice), 'std': np.std(all_dice)}
    if all_bce:
        results['bce'] = {'mean': np.mean(all_bce), 'std': np.std(all_bce)}
    if all_cnr:
        results['cnr'] = {'mean': np.mean(all_cnr), 'std': np.std(all_cnr)}

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        results_file = save_dir / config.eval_results_file
        with open(results_file, 'w') as f:
            for metric, values in results.items():
                f.write(f"{metric}: {values['mean']:.4f} ± {values['std']:.4f}\n")

    return results

if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader
    from config import get_config
    from FMT.FMT_dataset import load_fmtd_data
    from FMT.FMT_Models import get_model

    # Load configuration
    config = get_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load test data
    _, test_dataset = load_fmtd_data("DataSets", config.ssl_eval_type)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.eval_batch_size
    )

    # Load model
    model = get_model(config).to(device)
    model.load_state_dict(torch.load(config.model_path))

    # Evaluate
    results = evaluate_model(
        model, 
        test_loader, 
        device,
        config,
        save_dir=config.eval_save_dir
    )
    
    print("\nEvaluation Results:")
    for metric, values in results.items():
        print(f"{metric}: {values['mean']:.4f} ± {values['std']:.4f}")

    if config.eval_visualize:
        # Add visualization code here
        pass