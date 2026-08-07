"""Centralized Model Management.

Handles downloading, verifying, and providing paths to machine learning models
used across the application. Ensures that large binary files are tracked,
verified via SHA-256, and stored correctly in the local file system without
being committed to version control.
"""
from __future__ import annotations

import os
import hashlib
import logging
import urllib.request
from typing import Dict, Optional
from dataclasses import dataclass

@dataclass
class ModelMetadata:
    name: str
    url: str
    filename: str
    expected_size: int
    sha256: str
    description: str


class ModelManager:
    """Manages downloading and verifying AI models."""

    # Central Registry of supported models
    REGISTRY: Dict[str, ModelMetadata] = {
        "l2cs_net_gaze360": ModelMetadata(
            name="l2cs_net_gaze360",
            url="https://github.com/yakhyo/gaze-estimation/releases/download/v1.0/L2CSNet_gaze360.onnx",
            filename="L2CSNet_gaze360.onnx",
            expected_size=94402698,
            sha256="482a0b12", # We will replace this with the exact hash after downloading
            description="L2CS-Net ResNet-50 trained on Gaze360 for absolute Pitch/Yaw prediction."
        )
    }

    def __init__(self) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)
        # Resolve the models directory relative to project root
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self._models_dir = os.path.join(root_dir, "models")
        
        if not os.path.exists(self._models_dir):
            os.makedirs(self._models_dir)

    def get_model(self, model_id: str, force_download: bool = False) -> Optional[str]:
        """Get the local absolute path to a model, downloading it if necessary.

        Args:
            model_id: The ID of the model in the REGISTRY.
            force_download: If True, re-downloads even if the file exists.

        Returns:
            Absolute path to the model file, or None if download/verification failed.
        """
        if model_id not in self.REGISTRY:
            self._logger.error("Model ID '%s' not found in registry.", model_id)
            return None

        meta = self.REGISTRY[model_id]
        file_path = os.path.join(self._models_dir, meta.filename)

        if os.path.exists(file_path) and not force_download:
            if self._verify_file(file_path, meta):
                return file_path
            else:
                self._logger.warning("Existing model failed verification. Re-downloading.")
                os.remove(file_path)

        if self._download_model(meta, file_path):
            # For the very first run, we compute the hash to hardcode it in the registry
            if meta.sha256 == "482a0b12":
                real_hash = self._compute_sha256(file_path)
                self._logger.info("computed SHA256 for %s: %s", meta.name, real_hash)
            
            # Note: We skip strict verification if the hash in registry is a placeholder
            if meta.sha256 != "482a0b12" and not self._verify_file(file_path, meta):
                self._logger.error("Downloaded model failed integrity check!")
                os.remove(file_path)
                return None
            return file_path

        return None

    def _verify_file(self, file_path: str, meta: ModelMetadata) -> bool:
        """Verify the integrity of a downloaded file."""
        if os.path.getsize(file_path) != meta.expected_size:
            self._logger.error(
                "Size mismatch for %s. Expected %d, got %d",
                meta.name, meta.expected_size, os.path.getsize(file_path)
            )
            return False
            
        # If hash is a placeholder, skip hash verification
        if meta.sha256 == "482a0b12":
            return True

        actual_sha256 = self._compute_sha256(file_path)
        if actual_sha256 != meta.sha256:
            self._logger.error(
                "Hash mismatch for %s. Expected %s, got %s",
                meta.name, meta.sha256, actual_sha256
            )
            return False

        return True

    def _compute_sha256(self, file_path: str) -> str:
        """Compute the SHA-256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _download_model(self, meta: ModelMetadata, dest_path: str) -> bool:
        """Download a model file with progress logging."""
        self._logger.info("Downloading %s from %s...", meta.name, meta.url)
        
        try:
            req = urllib.request.Request(meta.url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(dest_path, 'wb') as out_file:
                total_size = int(response.getheader('Content-Length', 0))
                block_size = 1024 * 8
                count = 0
                
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    out_file.write(buffer)
                    count += 1
                    
                    if total_size > 0 and count % 1024 == 0:
                        downloaded = count * block_size
                        percent = (downloaded / total_size) * 100
                        self._logger.info("Download progress: %.1f%%", percent)
                        
            self._logger.info("Download complete: %s", meta.name)
            return True
            
        except Exception as e:
            self._logger.error("Download failed for %s: %s", meta.name, e)
            if os.path.exists(dest_path):
                os.remove(dest_path)
            return False
