import os
import json
import shutil
import random

def convert_coco_to_yolo(coco_json_path, dataset_dir, output_dir):
    with open(coco_json_path, 'r') as f:
        data = json.load(f)

    # Create mapping from category id to continuous id starting from 0
    categories = data['categories']
    cat_id_to_yolo_id = {cat['id']: i for i, cat in enumerate(categories)}
    yolo_id_to_name = {i: cat['name'] for i, cat in enumerate(categories)}

    # Group annotations by image_id
    img_to_anns = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)

    images = data['images']
    random.shuffle(images)

    # 80% train, 10% val, 10% test
    n = len(images)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    splits = {
        'train': images[:train_end],
        'val': images[train_end:val_end],
        'test': images[val_end:]
    }

    # Setup directories
    os.makedirs(output_dir, exist_ok=True)
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

    for split, split_images in splits.items():
        print(f"Processing {split} set...")
        for img in split_images:
            img_id = img['id']
            img_w = img['width']
            img_h = img['height']
            # Original file path in TACO is like 'batch_1/000006.jpg'
            orig_file_path = os.path.join(dataset_dir, img['file_name'])
            
            # Create a flat unique name
            new_file_name = img['file_name'].replace('/', '_')
            new_img_path = os.path.join(output_dir, 'images', split, new_file_name)
            
            # Copy image if exists
            if os.path.exists(orig_file_path):
                shutil.copy2(orig_file_path, new_img_path)
                # Write label file only if image exists
                txt_file_name = os.path.splitext(new_file_name)[0] + '.txt'
                txt_file_path = os.path.join(output_dir, 'labels', split, txt_file_name)

                with open(txt_file_path, 'w') as f_txt:
                    if img_id in img_to_anns:
                        for ann in img_to_anns[img_id]:
                            cat_id = ann['category_id']
                            yolo_id = cat_id_to_yolo_id[cat_id]
                            
                            # COCO bbox: [x_min, y_min, width, height]
                            bbox = ann['bbox']
                            x_min, y_min, w, h = bbox
                            
                            # YOLO bbox: [x_center, y_center, width, height] normalized
                            x_center = (x_min + w / 2.0) / img_w
                            y_center = (y_min + h / 2.0) / img_h
                            w_norm = w / img_w
                            h_norm = h / img_h
                            
                            # Ensure values are strictly between 0 and 1
                            x_center = max(0.0, min(1.0, x_center))
                            y_center = max(0.0, min(1.0, y_center))
                            w_norm = max(0.0, min(1.0, w_norm))
                            h_norm = max(0.0, min(1.0, h_norm))

                            f_txt.write(f"{yolo_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
            else:
                print(f"Warning: {orig_file_path} not found. Skipping.")

    # Generate dataset.yaml
    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as f_yaml:
        f_yaml.write(f"path: {os.path.abspath(output_dir).replace(chr(92), '/')}\n")
        f_yaml.write(f"train: images/train\n")
        f_yaml.write(f"val: images/val\n")
        f_yaml.write(f"test: images/test\n")
        f_yaml.write(f"\n")
        f_yaml.write(f"nc: {len(categories)}\n")
        f_yaml.write(f"names:\n")
        for i in range(len(categories)):
            f_yaml.write(f"  {i}: {yolo_id_to_name[i]}\n")
            
    print(f"Conversion complete. dataset.yaml saved to {yaml_path}")

if __name__ == '__main__':
    coco_json = 'dataset/annotations.json'
    dataset_dir = 'dataset'
    output_dir = 'dataset_yolo'
    convert_coco_to_yolo(coco_json, dataset_dir, output_dir)
