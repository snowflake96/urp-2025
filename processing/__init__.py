"""
NIfTI Processing Package
A comprehensive package for processing, validating, analyzing, and extracting aneurysms from NIfTI files.
"""

from .NIfTI_to_stl import nifti_to_stl
from .NIfTI_validation import validate_nifti, NIfTIValidator
from .NIfTI_find_aneurysm import find_aneurysms, AneurysmDetector
from .NIfTI_extract import extract_aneurysms, AneurysmExtractor

__version__ = "1.0.0"
__author__ = "URP-2025 Jiwoo Lee"

__all__ = [
    # Main functions
    'nifti_to_stl',
    'validate_nifti', 
    'find_aneurysms',
    'extract_aneurysms',
    
    # Classes
    'NIfTIValidator',
    'AneurysmDetector', 
    'AneurysmExtractor'
] 