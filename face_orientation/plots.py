import cv2
import numpy as np
import matplotlib.pyplot as plt

class OrientationPlots:
    def __init__(self):
        pass

    @staticmethod
    def plot_image_with_orientation(image, orientation, image_title):
        plt.figure(figsize=(6, 6))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image_rgb)
        plt.title(f"{image_title}\nPredicted Orientation: {orientation}°", fontsize=12)
        plt.axis('off')
        plt.show()

    @staticmethod
    def plot_edge_density_profiles(edges, image_title):
        vertical_profile = np.sum(edges, axis=1)
        horizontal_profile = np.sum(edges, axis=0)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # vertical profile
        axes[0].plot(vertical_profile, np.arange(len(vertical_profile)))
        axes[0].invert_yaxis()
        axes[0].set_title("Vertical Edge Density")
        axes[0].set_xlabel("Edge Density")
        axes[0].set_ylabel("Image Height (pixels)")

        # horizontal profile
        axes[1].plot(horizontal_profile)
        axes[1].set_title("Horizontal Edge Density")
        axes[1].set_xlabel("Image Width (pixels)")
        axes[1].set_ylabel("Edge Density")

        fig.suptitle(f"Edge Density Profiles for {image_title}", fontsize=14)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_orientation_analysis(image, edges, orientation, image_title):
        vertical_profile = np.sum(edges, axis=1)
        horizontal_profile = np.sum(edges, axis=0)

        # Ajuste na organização do GridSpec
        fig = plt.figure(figsize=(10, 8))
        gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[4, 1])

        # Imagem original (top-left, grande)
        ax_img = fig.add_subplot(gs[0, 0])
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax_img.imshow(image_rgb)
        ax_img.set_title(f"{image_title}\nPredicted Orientation: {orientation}°", fontsize=12)
        ax_img.axis('off')

        # vertical profile (top-right)
        ax_vprofile = fig.add_subplot(gs[0, 1])
        ax_vprofile.plot(vertical_profile, np.arange(len(vertical_profile)), color='blue')
        ax_vprofile.invert_yaxis()
        ax_vprofile.set_title("Vertical Edge Density")
        ax_vprofile.set_xlabel("Edge Density")
        ax_vprofile.set_ylabel("Height (pixels)")

        # horizontal profile (bottom-left)
        ax_hprofile = fig.add_subplot(gs[1, 0])
        ax_hprofile.plot(horizontal_profile, color='green')
        ax_hprofile.set_title("Horizontal Edge Density")
        ax_hprofile.set_xlabel("Width (pixels)")
        ax_hprofile.set_ylabel("Edge Density")

        # removes empty subplot (bottom-right)
        ax_empty = fig.add_subplot(gs[1, 1])
        ax_empty.axis('off')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_symmetry_analysis(image_gray, image_title):
        w = image_gray.shape[1]
        mid = w // 2
        left_half = image_gray[:, :mid]
        right_half = cv2.flip(image_gray[:, mid:], 1)
        diff = cv2.absdiff(left_half, right_half)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(left_half, cmap='gray')
        axes[0].set_title('Left Half')
        axes[0].axis('off')

        axes[1].imshow(right_half, cmap='gray')
        axes[1].set_title('Right Half (Mirrored)')
        axes[1].axis('off')

        axes[2].imshow(diff, cmap='hot')
        axes[2].set_title('Symmetry Difference')
        axes[2].axis('off')

        fig.suptitle(f"Symmetry Analysis for {image_title}", fontsize=14)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_skin_mask(image_bgr, image_title):
        image_ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
        lower = np.array([0, 133, 77], dtype=np.uint8)
        upper = np.array([255, 173, 127], dtype=np.uint8)
        skin_mask = cv2.inRange(image_ycrcb, lower, upper)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(skin_mask, cmap='gray')
        ax.set_title(f"Skin Mask for {image_title}", fontsize=12)
        ax.axis('off')
        plt.tight_layout()
        plt.show()