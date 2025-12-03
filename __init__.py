"""
Text Metrics Package
A toolkit for computing quantifiable text complexity and diversity metrics
"""

from .descriptor import TextDescriptor
from .metrics import TextMetrics
from .sentiment import SentimentMetrics

__version__ = "0.1.0"
__all__ = ["TextDescriptor", "TextMetrics", "SentimentMetrics"]
