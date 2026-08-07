"""Configuration manager for loading, saving, and resetting settings.

Persists AppSettings to a JSON file in a user-specific config directory.
All file I/O is wrapped in exception handling with logging.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from config.settings import AppSettings
from core.exceptions import ConfigError


class ConfigManager:
    """Handles loading and saving of application settings to JSON.

    Default config location: ~/.eye_tracking_control/config.json

    Attributes:
        settings: The current application settings (read-only property).
    """

    def __init__(self, config_dir: Optional[Path] = None) -> None:
        """Initialize the ConfigManager.

        Args:
            config_dir: Directory for the config file. Defaults to
                        ~/.eye_tracking_control/ if not specified.
        """
        self._logger = logging.getLogger(self.__class__.__name__)

        if config_dir is None:
            self._config_dir = Path.home() / ".eye_tracking_control"
        else:
            self._config_dir = Path(config_dir)

        self._config_path: Path = self._config_dir / "config.json"
        self._settings: AppSettings = AppSettings()

    @property
    def config_dir(self) -> Path:
        """Get the directory where configuration is stored."""
        return self._config_dir
        
    @property
    def calibration_dir(self) -> Path:
        """Get the directory where calibration datasets are stored."""
        d = self._config_dir / "datasets" / "calibration"
        d.mkdir(parents=True, exist_ok=True)
        return d
        
    @property
    def validation_dir(self) -> Path:
        """Get the directory where validation datasets are stored."""
        d = self._config_dir / "datasets" / "validation"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def load(self) -> AppSettings:
        """Load settings from the JSON config file.

        If the file does not exist, creates it with default settings.
        If the file contains invalid JSON, raises ConfigError.
        Missing keys are filled with defaults.

        Returns:
            The loaded (or default) AppSettings.

        Raises:
            ConfigError: If the config file cannot be read or parsed.
        """
        if not self._config_path.exists():
            self._logger.info(
                "Config file not found at %s. Creating with defaults.",
                self._config_path,
            )
            self.save()
            return self._settings

        try:
            with open(self._config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._settings = AppSettings.from_dict(data)
            self._logger.debug("Loaded config from %s", self._config_path)
        except json.JSONDecodeError as e:
            self._logger.error("Invalid JSON in config file: %s", e)
            raise ConfigError(f"Failed to parse config file: {e}") from e
        except OSError as e:
            self._logger.error("Error reading config file: %s", e)
            raise ConfigError(f"Failed to read config: {e}") from e

        return self._settings

    def save(self, settings: Optional[AppSettings] = None) -> None:
        """Save settings to the JSON config file.

        Args:
            settings: Settings to save. If None, saves the current settings.

        Raises:
            ConfigError: If the config file cannot be written.
        """
        if settings is not None:
            self._settings = settings

        try:
            self._config_dir.mkdir(parents=True, exist_ok=True)
            with open(self._config_path, "w", encoding="utf-8") as f:
                json.dump(self._settings.to_dict(), f, indent=4)
            self._logger.debug("Saved config to %s", self._config_path)
        except OSError as e:
            self._logger.error("Error saving config file: %s", e)
            raise ConfigError(f"Failed to save config: {e}") from e

    def reset(self) -> AppSettings:
        """Reset settings to defaults, save, and return them.

        Returns:
            The default AppSettings.
        """
        self._logger.info("Resetting configuration to defaults.")
        self._settings = AppSettings()
        self.save()
        return self._settings

    @property
    def settings(self) -> AppSettings:
        """Get the current application settings."""
        return self._settings

    def update(self, **kwargs: object) -> None:
        """Update specific top-level settings attributes.

        Args:
            **kwargs: Attribute names and values to update.
                      Unknown attributes are logged and ignored.
        """
        for key, value in kwargs.items():
            if hasattr(self._settings, key):
                setattr(self._settings, key, value)
            else:
                self._logger.warning(
                    "Attempted to update unknown setting: %s", key
                )
