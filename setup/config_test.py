from movie_style.feature.depth_estimator import DepthEstimator
from movie_style.feature.feature_mixer import FeatureMixer
from movie_style.feature.focus_detector import FocusDetector
from movie_style.feature.layer_detector import LayerDetector
from movie_style.feature.motion_detector import MotionDetector
from movie_style.feature.people_detector import PeopleDetector
from movie_style.feature.pose_estimator import PoseEstimator
from movie_style.feature.toric_estimator import ToricEstimator
from movie_style.model.naive_model import NaiveMLP


def test_config():
    """Test the environment configuration by initializing main classes."""
    DepthEstimator()
    FeatureMixer()
    FocusDetector()
    LayerDetector()
    MotionDetector()
    PeopleDetector()
    PoseEstimator()
    ToricEstimator()
    NaiveMLP([1, 1])


if __name__ == "__main__":
    test_config()
    print("The environment seems well configured")
