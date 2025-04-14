import cv2
import numpy as np
import pandas as pd
from face_orientation.plots import OrientationPlots

class OrientationAnalyzer:
    def __init__(self, plot_full_analysis=True, plot_results=False, plot_histograms=False):
        self.plot_full_analysis = plot_full_analysis
        self.plot_results = plot_results
        self.plot_histograms = plot_histograms

    @staticmethod
    def preprocess(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0)
        return clahe.apply(gray)

    @staticmethod
    def edge_density_profile(edges):
        vertical_profile = np.sum(edges, axis=1)
        horizontal_profile = np.sum(edges, axis=0)
        return vertical_profile, horizontal_profile
    
    @staticmethod
    def symmetry_score(image_gray):
        h, w = image_gray.shape
        mid = w // 2
        left_half = image_gray[:, :mid]
        right_half = cv2.flip(image_gray[:, mid:], 1)  # mirrors half left portion of image

        # calculate absolute diff between parts
        symmetry = cv2.absdiff(left_half, right_half)
        score = -np.mean(symmetry)  # the lower the diff, the bigger the simmetry (higher scores)
        return score

    @staticmethod
    def skin_tone_score(image_bgr):
        # convert to YCrCb (more stable for skin tones)
        image_ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)

        # usual ranges for skin tones
        lower = np.array([0, 133, 77], dtype=np.uint8)
        upper = np.array([255, 173, 127], dtype=np.uint8)

        # skin tone mask
        skin_mask = cv2.inRange(image_ycrcb, lower, upper)

        # mask vertical profile
        vertical_profile = np.sum(skin_mask, axis=1)

        h = len(vertical_profile)
        top = vertical_profile[:h//3].mean()
        middle = vertical_profile[h//3:2*h//3].mean()
        bottom = vertical_profile[2*h//3:].mean()

        score = top * 1.2 + middle - bottom  # more skin in the upper half = higher score
        return score


    @staticmethod
    def orientation_score(vertical_profile):
        h = len(vertical_profile)
        top = vertical_profile[:h//3].mean()
        middle = vertical_profile[h//3:2*h//3].mean()
        bottom = vertical_profile[2*h//3:].mean()

        score = top * 1.5 + middle - bottom
        return score

    @staticmethod    
    def normalize_scores(scores):
        scores = np.array(scores, dtype=np.float32)
        if scores.max() == scores.min():
            return np.ones_like(scores)  # to prevent from divisions by zero
        return (scores - scores.min()) / (scores.max() - scores.min())

    @staticmethod
    def rotate_image(image, angle):
        rotations = {
            0: image,
            90: cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE),
            180: cv2.rotate(image, cv2.ROTATE_180),
            270: cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        }
        return rotations.get(angle, image)

    def analyze_orientation(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return None

        orientations = [0, 90, 180, 270]

        edge_scores = []
        sym_scores = []
        skin_scores = []

        for angle in orientations:
            rotated = self.rotate_image(image, angle)
            preprocessed = self.preprocess(rotated)
            edges = cv2.Canny(preprocessed, 100, 200)

            # score based on vertical profile of edges
            vertical_profile, _ = self.edge_density_profile(edges)
            edge_score = self.orientation_score(vertical_profile)
            edge_scores.append(edge_score)

            # symmetry score
            sym_score = self.symmetry_score(preprocessed)
            sym_scores.append(sym_score)

            # skin tone concentration score
            skin_score = self.skin_tone_score(rotated)
            skin_scores.append(skin_score)

        # normalizing scores
        edge_scores_norm = self.normalize_scores(edge_scores)
        sym_scores_norm = self.normalize_scores(sym_scores)
        skin_scores_norm = self.normalize_scores(skin_scores)

        df_scores = pd.DataFrame(index=orientations)
        df_scores['edge_scores'] = edge_scores_norm
        df_scores['sym_scores'] = sym_scores_norm
        df_scores['skin_scores'] = skin_scores_norm
        df_scores['combined'] = df_scores.apply(lambda x: x['edge_scores'] + 0.15 * x['sym_scores'] + 0.1 * x['skin_scores'], axis=1)

        # print(df_scores)
        scores = df_scores.combined.to_list()

        best_orientation = orientations[np.argmax(scores)]
        print(f"Image: {image_path}, Estimated orientation: {best_orientation} degrees")

        if self.plot_full_analysis:
            plotter = OrientationPlots()
            plotter.plot_orientation_analysis(image, edges, best_orientation, image_path.split("/")[-1])

        if self.plot_results:
            plotter = OrientationPlots()
            plotter.plot_image_with_orientation(image, best_orientation, image_title=image_path.split("/")[-1])

        if self.plot_histograms:
            plotter = OrientationPlots()
            plotter.plot_symmetry_analysis(image, image_path.split("/")[-1])
            plotter.plot_skin_mask(image, image_path.split("/")[-1])

        return best_orientation
