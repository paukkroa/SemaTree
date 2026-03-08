"""Local filesystem crawler for markdown, text, RST, and HTML files."""

from __future__ import annotations

import logging
from pathlib import Path

from markdownify import markdownify

from sema_tree.crawlers.base import CrawledPage
from sema_tree.models import RefType

logger = logging.getLogger(__name__)

# Patterns typically found in .gitignore that we always skip.
_ALWAYS_SKIP = {".git", "__pycache__", "node_modules", ".venv", "venv", ".tox", ".mypy_cache"}


class LocalCrawler:
    """Walk a directory tree and collect document files.

    Respects ``.gitignore`` patterns found at the root of *path* and skips
    common non-content directories.
    """

    def __init__(
        self,
        path: str,
        extensions: tuple[str, ...] = (".md", ".txt", ".rst", ".html"),
    ) -> None:
        self.root = Path(path).resolve()
        self.extensions = extensions

    async def crawl(self) -> list[CrawledPage]:
        if not self.root.is_dir():
            raise FileNotFoundError(f"Directory not found: {self.root}")

        ignore_patterns = self._load_gitignore()
        pages: list[CrawledPage] = []

        for file_path in sorted(self.root.rglob("*")):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in self.extensions:
                continue
            if self._is_ignored(file_path, ignore_patterns):
                continue

            try:
                content = file_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                logger.debug("Skipping %s: %s", file_path, exc)
                continue

            if file_path.suffix.lower() == ".html":
                content = self._html_to_markdown(content)

            title = file_path.stem.replace("-", " ").replace("_", " ").title()
            rel_path = file_path.relative_to(self.root)

            pages.append(
                CrawledPage(
                    url_or_path=str(file_path),
                    title=title,
                    content=content,
                    ref_type=RefType.file,
                    metadata={"relative_path": str(rel_path)},
                )
            )

        logger.info("Local crawl found %d files under %s", len(pages), self.root)
        return pages

    # ------------------------------------------------------------------
    # .gitignore handling
    # ------------------------------------------------------------------

    def _load_gitignore(self) -> list[str]:
        """Load gitignore patterns from the root directory."""
        gitignore_path = self.root / ".gitignore"
        if not gitignore_path.is_file():
            return []
        patterns: list[str] = []
        for line in gitignore_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            patterns.append(line)
        return patterns

    def _is_ignored(self, file_path: Path, patterns: list[str]) -> bool:
        """Check if a path should be ignored based on gitignore patterns."""
        rel = file_path.relative_to(self.root)

        # Always skip well-known non-content directories.
        for part in rel.parts:
            if part in _ALWAYS_SKIP:
                return True

        rel_str = str(rel)
        for pattern in patterns:
            clean = pattern.rstrip("/")
            # Directory-only pattern (ends with /).
            if pattern.endswith("/"):
                if any(part == clean for part in rel.parts[:-1]):
                    return True
                if rel.parts and rel.parts[0] == clean:
                    return True
                continue

            # Exact filename or directory name match at any level.
            if "/" not in clean:
                if file_path.name == clean:
                    return True
                if any(part == clean for part in rel.parts):
                    return True
                # Simple glob: *.ext
                if clean.startswith("*.") and file_path.suffix == clean[1:]:
                    return True
                continue

            # Path-based pattern: match from root.
            if rel_str == clean or rel_str.startswith(clean + "/"):
                return True

        return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _html_to_markdown(html: str) -> str:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        main = soup.find("main") or soup.find("article") or soup.find("body") or soup
        return markdownify(str(main), heading_style="ATX", strip=["img"]).strip()
