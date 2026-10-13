"""Colab storage contract: resolve output paths inside an active Drive mount."""
import os
from pathlib import Path

DEFAULT_MOUNT = '/content/drive'
DEFAULT_OUTPUT = '/content/drive/MyDrive/simulations'

def require_drive_path(path, mount=None):
    mount = Path(mount or os.environ.get('GPE_DRIVE_MOUNT', DEFAULT_MOUNT)).expanduser().resolve()
    if not os.path.ismount(mount) or not (mount / 'MyDrive').is_dir():
        raise ValueError(f'Google Drive is not mounted at {mount}. Mount it in the notebook before preparing or writing a run.')
    path = Path(path).expanduser().resolve()
    roots = (mount / 'MyDrive', mount / 'Shareddrives')
    if not any(path.is_relative_to(root) and path != root for root in roots):
        raise ValueError('Choose a simulation folder inside mounted Google Drive, not temporary runtime storage.')
    if 'GPE_simulator' in path.parts:
        raise ValueError('GPE_simulator is read-only reference material; choose a separate simulations folder.')
    return path

def check_storage(path):
    # Set by the Colab runner and launcher. CPU-only I/O tests need no mounted Drive.
    if os.environ.get('GPE_DRIVE_MOUNT'):
        require_drive_path(path)
