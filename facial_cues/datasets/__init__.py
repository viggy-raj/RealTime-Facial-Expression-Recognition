"""
Datasets package — loaders for CK+, CASME II, SAMM.
Imports are lazy to avoid failures when cv2/torch are not installed.
"""
try:
    from .ck_plus_dataset import CKPlusDataset, CKPlusSequenceDataset
except Exception:
    pass  # cv2 or torch not installed yet
