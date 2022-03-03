import os
import json
import shutil 
from PIL import Image as pimg

folders = ['00/', '01/', '02/', '03/', '04/', '05/']

f = json.load(open('annotations.json', 'r'))

annotations = f['annotations']
images = {}


for i,an in enumerate(annotations):
    id = an['image_id']

    if id not in images:
        images[id] = (i,0)
    ind, num = images[id]
    images[id] = (ind, num+1)

single_person = {}

for k,v in images.items():
    if v[1] == 1:
        single_person[k] = annotations[v[0]]

good_dimensions = {}

threshold = 0.05 * 640 * 480

for k,v in single_person.items():
    if v['area'] > threshold and v['num_keypoints'] == 14:
        good_dimensions[k] = v

file_location = {}
for fld in folders:
    for filename in os.listdir(fld):
        if 'rgb' not in filename:
            continue
        file_location[filename] = fld

clean_dataset_path = 'clean_dataset/'
if os.path.exists(clean_dataset_path):
    shutil.rmtree(clean_dataset_path)

os.makedirs(os.path.dirname(clean_dataset_path))
for k,v in file_location.items():
    im_id = int(k.split('.')[0])
    if im_id in good_dimensions:
        shutil.copy(v + k, clean_dataset_path + k)

with open('clean_annotations.json', 'w') as fout:
    json.dump(good_dimensions, fout)


for filename in os.listdir(clean_dataset_path):
    img = pimg.open(clean_dataset_path + filename)
    img.load()
    background = pimg.new("RGB", img.size, (255, 255, 255))
    background.paste(img, mask = img.split()[3])
    background.save(clean_dataset_path + filename, "PNG", quality=300)


