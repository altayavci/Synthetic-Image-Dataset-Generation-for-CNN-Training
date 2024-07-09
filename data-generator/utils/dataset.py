import pandas as pd
import os


def get_images(path: str, sample_dataset=None):
    imgs = [img for img in os.listdir(path) if ".ipynb" not in img]
    if sample_dataset:
        df = pd.DataFrame(imgs)
        df = df.sample(frac=sample_dataset)
        imgs = df.values.flatten().tolist()
    return imgs
