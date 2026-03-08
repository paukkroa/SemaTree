"""Crawlers for extracting content from various sources."""

from .base import CrawledPage, Crawler
from .local import LocalCrawler
from .web import WebCrawler

__all__ = ["CrawledPage", "Crawler", "LocalCrawler", "WebCrawler"]
