import cv2
import os

def generate_rotated_images(image_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    new_image_paths = []

    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error trying to load the image: {image_path}")
            continue

        base_name = os.path.splitext(os.path.basename(image_path))[0]

        # rotate 90 degrees
        rotated_90 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        path_90 = os.path.join(output_dir, f"{base_name}_rot90.jpg")
        cv2.imwrite(path_90, rotated_90)
        new_image_paths.append(path_90)

        # rotate -90 degrees (or 270 degrees)
        rotated_minus_90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        path_minus_90 = os.path.join(output_dir, f"{base_name}_rot-90.jpg")
        cv2.imwrite(path_minus_90, rotated_minus_90)
        new_image_paths.append(path_minus_90)

        # rotate 180 degrees
        rotated_180 = cv2.rotate(image, cv2.ROTATE_180)
        path_180 = os.path.join(output_dir, f"{base_name}_rot180.jpg")
        cv2.imwrite(path_180, rotated_180)
        new_image_paths.append(path_180)

        print(f"Rotated images were successfully generated for {base_name}")

    return list(set(image_paths + new_image_paths))
