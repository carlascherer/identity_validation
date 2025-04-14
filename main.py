from utils.image_io import load_images
from face_orientation.orientation_analyzer import OrientationAnalyzer

def main():
    input_dir = 'data'

    images = load_images(input_dir=input_dir)

    analyzer = OrientationAnalyzer(plot_full_analysis=False, plot_histograms=False)
    results = {}
    for image_path in images:#[:5]:
        orientation = analyzer.analyze_orientation(image_path)
        results[image_path] = orientation

    print("\nOrientation Summary:")
    for path, orientation in results.items():
        print(f"{path}: {orientation} degrees")

if __name__ == "__main__":
    main()
