from pathlib import Path
import torch 
from torchvision import utils 
import math 
from NIR_II_Guided_Diffusion.models.pipelines import DiffusionPipeline
from NIR_II_Guided_Diffusion.utils.general_utils import normalization
from config import get_config


if __name__ == "__main__":
    # ------------ Config --------------------
    config = get_config()
    
    path_out = Path(config.results_dir) 
    path_out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(config.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ------------ Load Model ------------
    pipeline = DiffusionPipeline.load_from_checkpoint('__your__diffusion_model__checkpoint__path__', strict=False)
    pipeline.to(device)
    
    # --------- Generate Samples  -------------------
    images = {}

    for cond in [0, None]:
        torch.manual_seed(config.seed)
 
        # --------- Conditioning ---------
        condition = torch.tensor([cond]*config.n_samples, device=device) if cond is not None else None 
        un_cond = None 

        # ----------- Run --------
        results = pipeline.sample(
            config.n_samples, 
            config.sample_size,
            guidance_scale=config.guidance_scale,
            condition=condition,
            un_cond=un_cond,
            steps=config.sampling_steps,
            use_ddim=config.use_ddim
        )

        # --------- Save result ---------------
        if config.save_normalized:
            results = (results + 1) / 2  # Transform from [-1, 1] to [0, 1]
            results = results.clamp(0, 1)
            
        utils.save_image(
            results, 
            path_out / f'test_{cond}.png',
            nrow=int(math.sqrt(results.shape[0])),
            normalize=True,
            scale_each=True
        )
        images[cond] = results


