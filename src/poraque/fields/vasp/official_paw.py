"""Compatibility alias for the runtime-only user-provided PAW parser.

A valid VASP license is required. No potential data is included, generated or
located automatically. This alias does not certify dataset authenticity.
"""
from .user_paw import UserPawDataset, PotcarLicenseWarning

OfficialPawDataset = UserPawDataset
__all__ = ['UserPawDataset', 'OfficialPawDataset', 'PotcarLicenseWarning']
