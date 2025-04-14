from utils.image_io import download_and_sample_images
from utils.image_rotation import generate_rotated_images

def generate_input_images():
    original_dir = "data/original"
    rotated_dir = "data/rotated"

    images = download_and_sample_images(n=20, output_dir=original_dir)
    all_images = generate_rotated_images(images, rotated_dir)

    return all_images


if __name__ == "__generate_input_images__":
    generate_input_images()
