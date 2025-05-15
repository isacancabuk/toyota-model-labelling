# split.py
from pathlib import Path
import random, shutil

SRC = Path('data/4runner')
TRAIN = Path('data/4runner_split/train')
VAL   = Path('data/4runner_split/val')
TRAIN.mkdir(parents=True, exist_ok=True)
VAL.mkdir(parents=True, exist_ok=True)

imgs = list(SRC.glob('*.jpg'))
random.shuffle(imgs)
cut = int(0.8 * len(imgs))
for i,img in enumerate(imgs):
    tgt = TRAIN if i<cut else VAL
    # copy image
    shutil.copy(img, tgt/img.name)
    # copy label if varsa
    lbl = img.with_suffix('.txt')
    if lbl.exists():
        shutil.copy(lbl, tgt/lbl.name)
print("Split tamam.")