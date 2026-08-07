import importlib
import pkgutil
import inspect
import logging
from typing import Dict, Type
from .interfaces import BaseRetriever, BaseEmbedder, BaseGenerator, BaseEvaluator

logger = logging.getLogger(__name__)

class PluginRegistry:
    def __init__(self):
        self.retrievers: Dict[str, Type[BaseRetriever]] = {}
        self.embedders: Dict[str, Type[BaseEmbedder]] = {}
        self.generators: Dict[str, Type[BaseGenerator]] = {}
        self.evaluators: Dict[str, Type[BaseEvaluator]] = {}

    def discover_plugins(self, package_name: str = "backend.plugins"):
        try:
            package = importlib.import_module(package_name)
        except ImportError as e:
            logger.warning(f"Plugin package '{package_name}' not found: {e}")
            return

        for _, module_name, is_pkg in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
            try:
                module = importlib.import_module(module_name)
                self._register_from_module(module)
            except Exception as e:
                logger.error(f"Failed to load plugin module {module_name}: {e}")

    def _register_from_module(self, module):
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and not inspect.isabstract(obj):
                if issubclass(obj, BaseRetriever) and obj is not BaseRetriever:
                    self.retrievers[name] = obj
                    logger.info(f"Registered Retriever plugin: {name}")
                elif issubclass(obj, BaseEmbedder) and obj is not BaseEmbedder:
                    self.embedders[name] = obj
                    logger.info(f"Registered Embedder plugin: {name}")
                elif issubclass(obj, BaseGenerator) and obj is not BaseGenerator:
                    self.generators[name] = obj
                    logger.info(f"Registered Generator plugin: {name}")
                elif issubclass(obj, BaseEvaluator) and obj is not BaseEvaluator:
                    self.evaluators[name] = obj
                    logger.info(f"Registered Evaluator plugin: {name}")

registry = PluginRegistry()
