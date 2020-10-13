import os
from PIL import Image

base = './BAM'
folders = sorted(os.listdir(base))

for folder in folders:
    if folder == '.DS_Store': continue
    imgs = os.listdir(os.path.join(base, folder))

    for img in imgs:
        img_path = os.path.join(base, folder, img)
        im = Image.open(img_path)
        if im.size[1] >= 5000:
            print("remove")
            os.remove(img_path)
