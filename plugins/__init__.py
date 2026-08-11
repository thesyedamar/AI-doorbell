"""
plugins/__init__.py — Plugin Loader
=====================================
Discovers and loads enabled plugins at startup.
"""

import importlib
import logging
from plugins.base import DoorbellPlugin, VisitorEvent

logger = logging.getLogger("plugins.loader")


class PluginLoader:
    """Loads, holds, and dispatches events to all active plugins."""

    def __init__(self):
        self._plugins: list = []

    def load_all(self, enabled_names: list) -> None:
        for name in enabled_names:
            if not name:
                continue
            try:
                module = importlib.import_module(f"plugins.{name}")
                plugin_class = None
                for attr in vars(module).values():
                    if (isinstance(attr, type)
                            and issubclass(attr, DoorbellPlugin)
                            and attr is not DoorbellPlugin):
                        plugin_class = attr
                        break
                if plugin_class is None:
                    logger.warning(f"Plugin '{name}' has no DoorbellPlugin subclass — skipped.")
                    continue
                plugin = plugin_class()
                plugin.on_load()
                self._plugins.append(plugin)
                logger.info(f"Loaded plugin: {name} v{plugin.version}")
            except Exception as e:
                logger.error(f"Failed to load plugin '{name}': {e}")

    def unload_all(self) -> None:
        for plugin in self._plugins:
            try:
                plugin.on_unload()
            except Exception as e:
                logger.error(f"Error unloading plugin '{plugin.name}': {e}")
        self._plugins.clear()

    def dispatch(self, hook: str, *args, **kwargs) -> None:
        for plugin in self._plugins:
            try:
                method = getattr(plugin, hook, None)
                if method:
                    method(*args, **kwargs)
            except Exception as e:
                logger.error(f"Plugin '{plugin.name}' error in {hook}: {e}")

    @property
    def count(self) -> int:
        return len(self._plugins)


# Module-level loader instance
loader = PluginLoader()