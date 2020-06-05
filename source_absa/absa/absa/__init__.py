from absa.absa import Predictor
from absa.absa import evaluate
from absa.absa import train
from absa.absa import suggest
import pkg_resources

__version__ = pkg_resources.get_distribution('absa').version
__all__ = ['Predictor', 'evaluate', 'train', 'suggest']
