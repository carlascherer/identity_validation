import os
import shutil
import random
import kagglehub

def download_and_sample_images(n=10, output_dir="data"):
    if not os.listdir('data/original'): 
        dataset_path = kagglehub.dataset_download("tapakah68/selfies-id-images-dataset")

        all_images = []
        for root, _, files in os.walk(dataset_path):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    all_images.append(os.path.join(root, f))

        # select N random images
        sample_images = random.sample(all_images, min(n, len(all_images)))

        # create output dir
        os.makedirs(output_dir, exist_ok=True)

        # copy images to output dir
        copied_paths = []
        for src_path in sample_images:
            filename = os.path.basename(src_path)
            dest_path = os.path.join(output_dir, filename)
            shutil.copy2(src_path, dest_path)
            copied_paths.append(dest_path)

        print(f"{len(copied_paths)} images copied to: {output_dir}")

        return copied_paths

def load_images(input_dir="data"):
    all_images = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_images.append(os.path.join(root, f))
    return list(set(all_images))

def delete_images_from_dir(image_paths):
    for path in image_paths:
        if os.path.exists(path):
            os.remove(path)
            print(f"Removed image: {path}")
