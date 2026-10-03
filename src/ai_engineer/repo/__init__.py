"""Repository intelligence: file listing, discovery, symbols, import graph and search."""

from __future__ import annotations

from ai_engineer.repo.discovery import (
    CommandSuggestion,
    ProjectProfile,
    discover,
    load_profile,
    save_profile,
)
from ai_engineer.repo.files import (
    DEFAULT_IGNORE_DIRS,
    is_generated_content,
    is_generated_path,
    is_git_repo,
    is_test_path,
    language_of,
    list_files,
)
from ai_engineer.repo.index import FileRecord, IndexStats, RepoIndex
from ai_engineer.repo.search import TextMatch, find_files, search_text
from ai_engineer.repo.symbols import Symbol, extract_imports, extract_symbols

__all__ = [
    "DEFAULT_IGNORE_DIRS",
    "CommandSuggestion",
    "FileRecord",
    "IndexStats",
    "ProjectProfile",
    "RepoIndex",
    "Symbol",
    "TextMatch",
    "discover",
    "extract_imports",
    "extract_symbols",
    "find_files",
    "is_generated_content",
    "is_generated_path",
    "is_git_repo",
    "is_test_path",
    "language_of",
    "list_files",
    "load_profile",
    "save_profile",
    "search_text",
]
