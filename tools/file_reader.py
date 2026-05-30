"""
AstraMind AI - File Reader Tool
=================================
A tool for reading and parsing various file formats.
Supports text files, CSV, JSON, and basic binary file metadata.
"""

import csv
import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class FileReaderTool:
    """
    A tool for reading and parsing files.

    Supports:
    - Text files (.txt, .md, .log)
    - CSV files (.csv)
    - JSON files (.json)
    - File metadata extraction
    - Content preview and summarization
    """

    name = "file_reader"
    description = "Reads and parses various file formats including text, CSV, and JSON."

    SUPPORTED_EXTENSIONS = {
        ".txt", ".md", ".log", ".csv", ".json", ".xml", ".yaml", ".yml",
        ".py", ".js", ".ts", ".html", ".css", ".env", ".ini", ".cfg",
    }

    MAX_PREVIEW_CHARS = 5000
    MAX_CSV_ROWS = 100

    async def execute(self, query: str) -> Dict[str, Any]:
        """
        Execute file reading based on the query.

        Args:
            query: A string containing a file path or file-related request.

        Returns:
            Dictionary with file contents and metadata.
        """
        logger.info(f"File reader processing: {query}")

        file_path = self._extract_file_path(query)

        if not file_path:
            return {
                "success": False,
                "error": "Tidak dapat menemukan path file dalam query.",
                "query": query,
            }

        if not os.path.exists(file_path):
            return {
                "success": False,
                "error": f"File tidak ditemukan: {file_path}",
                "file_path": file_path,
            }

        try:
            file_info = self._get_file_info(file_path)
            content = await self._read_file(file_path)

            return {
                "success": True,
                "file_path": file_path,
                "file_info": file_info,
                "content": content,
            }
        except Exception as e:
            logger.error(f"File read error: {e}")
            return {
                "success": False,
                "error": f"Kesalahan membaca file: {str(e)}",
                "file_path": file_path,
            }

    def _extract_file_path(self, query: str) -> Optional[str]:
        """
        Extract a file path from the query string.

        Looks for common file path patterns in the input.
        """
        import re

        # Pattern for absolute and relative file paths
        path_patterns = [
            r'(?:file|baca|read|open|buka)\s+[`"\']?([^\s`"\']+\.[a-zA-Z0-9]+)[`"\']?',
            r'([/\w\-\.]+\.[a-zA-Z0-9]+)',
            r'([\w\-\.]+\.[a-zA-Z0-9]+)',
        ]

        for pattern in path_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                potential_path = match.group(1)
                if any(potential_path.endswith(ext) for ext in self.SUPPORTED_EXTENSIONS):
                    return potential_path

        return None

    def _get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get metadata about a file."""
        stat = os.stat(file_path)
        _, ext = os.path.splitext(file_path)

        return {
            "name": os.path.basename(file_path),
            "extension": ext,
            "size_bytes": stat.st_size,
            "size_readable": self._format_size(stat.st_size),
            "modified_time": stat.st_mtime,
            "is_supported": ext in self.SUPPORTED_EXTENSIONS,
        }

    async def _read_file(self, file_path: str) -> Any:
        """
        Read a file based on its extension.

        Uses the appropriate parser for each file type.
        """
        _, ext = os.path.splitext(file_path).lower()

        if ext == ".csv":
            return self._read_csv(file_path)
        elif ext == ".json":
            return self._read_json(file_path)
        else:
            return self._read_text(file_path)

    def _read_text(self, file_path: str) -> Dict[str, Any]:
        """Read a text file with preview support."""
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        preview = content[:self.MAX_PREVIEW_CHARS]
        truncated = len(content) > self.MAX_PREVIEW_CHARS

        return {
            "type": "text",
            "preview": preview,
            "full_length": len(content),
            "truncated": truncated,
            "line_count": content.count("\n") + 1,
        }

    def _read_csv(self, file_path: str) -> Dict[str, Any]:
        """Read and parse a CSV file."""
        rows = []
        headers = []

        with open(file_path, "r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            for i, row in enumerate(reader):
                if i >= self.MAX_CSV_ROWS:
                    break
                rows.append(dict(row))

        return {
            "type": "csv",
            "headers": headers,
            "rows": rows,
            "row_count": len(rows),
            "column_count": len(headers),
            "truncated": len(rows) >= self.MAX_CSV_ROWS,
        }

    def _read_json(self, file_path: str) -> Dict[str, Any]:
        """Read and parse a JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return {
            "type": "json",
            "data": data,
            "data_type": type(data).__name__,
            "size": len(data) if isinstance(data, (list, dict)) else None,
        }

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """Format file size in human-readable format."""
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"
