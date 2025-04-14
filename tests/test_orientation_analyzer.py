import pytest
from face_orientation.orientation_analyzer import OrientationAnalyzer

@pytest.fixture
def analyzer():
    return OrientationAnalyzer()

@pytest.mark.parametrize("image_path,expected_orientation", [
    ("data/original/ID_1.jpg", 0)
    # ,(.....)
])
def test_analyze_orientation(analyzer, image_path, expected_orientation):
    orientation = analyzer.analyze_orientation(image_path, plot_result=False)
    assert orientation == expected_orientation, f"Expected {expected_orientation}, got {orientation}"
