from pathlib import Path
import torch 
from torchvision import utils 
import math 
from NIR_II_Guided_Diffusion.models.pipelines import DiffusionPipeline
import numpy as np 
from PIL import Image
import time
from tqdm import tqdm
from config import get_config
from NIR_II_Guided_Diffusion.utils.general_utils import chunks


if __name__ == "__main__":
    # ------------ Config --------------------
    config = get_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ------------ Load Model ------------
    pipeline = DiffusionPipeline.load_from_checkpoint(
        '__your__diffusion_model__checkpoint__path__', 
        strict=False
    ).to(device)

    # ------------ Generate Samples ------------
    for steps in config.sample_steps:
        for name, label in config.dataset_labels.items():
            # Setup output directory
            path_out = Path(config.output_dir) / f'generated_diffusion3_{steps}' / name
            path_out.mkdir(parents=True, exist_ok=True)

            # Set random seed
            torch.manual_seed(config.seed)
            
            # Initialize counter
            counter = 0
            
            # Generate samples in batches with progress bar
            for chunk in tqdm(
                chunks(list(range(config.samples_per_class)), config.batch_size),
                desc=f"Generating {name} samples for {steps} steps"
            ):
                # Setup conditioning
                condition = torch.tensor([label]*len(chunk), device=device) if label is not None else None 
                un_cond = torch.tensor([1-label]*len(chunk), device=device) if label is not None else None
                
                # Generate samples
                results = pipeline.sample(
                    len(chunk), 
                    config.img_size,
                    guidance_scale=config.cfg_scale,
                    condition=condition,
                    un_cond=un_cond,
                    steps=steps
                )

                results = results.cpu().numpy()
                
                # Save generated images
                for image in results:
                    # Normalize and convert to uint8
                    image = image.clip(-1, 1)
                    image = image * 255
                    image = np.moveaxis(image, 0, -1)
                    image = image.astype(np.uint8)
                    image = np.squeeze(image, axis=-1) if image.shape[-1]==1 else image
                    
                    # Save image
                    Image.fromarray(image).convert("RGB").save(
                        path_out / f'fake_{counter}.{config.save_format.lower()}'
                    )
                    counter += 1

            time.sleep(1)
