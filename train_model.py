from fastai.vision.all import *
from pathlib import Path

# Path to your dataset
path = Path("fossils")

# Load data
dls = ImageDataLoaders.from_folder(
    path,
    valid_pct=0.2,
    seed=42,
    item_tfms=Resize(224),
    batch_tfms=aug_transforms()
)

# Create learner with a pretrained model
learn = vision_learner(dls, resnet34, metrics=accuracy)

# Train the model
learn.fine_tune(4)

# Export model for inference
learn.export("model.pkl")