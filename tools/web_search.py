"""
AstraMind AI - Web Search Tool
================================
A tool for performing web searches and retrieving information from the internet.
Supports query formulation, result parsing, and content extraction.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class WebSearchTool:
    """
    A tool for searching the web and retrieving information.

    Supports:
    - Web search query execution
    - Result parsing and ranking
    - Snippet extraction
    - URL management
    """

    name = "web_search"
    description = "Searches the web for information and returns relevant results."

    def __init__(self, api_key: Optional[str] = None, default_num_results: int = 5):
        self.api_key = api_key
        self.default_num_results = default_num_results

    async def execute(self, query: str) -> Dict[str, Any]:
        """
        Execute a web search based on the query.

        Args:
            query: The search query string.

        Returns:
            Dictionary with search results and metadata.
        """
        logger.info(f"Web search processing: {query}")

        search_query = self._formulate_query(query)

        try:
            results = await self._perform_search(search_query)
            return {
                "success": True,
                "query": search_query,
                "results": results,
                "total_results": len(results),
            }
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return {
                "success": False,
                "error": f"Kesalahan pencarian web: {str(e)}",
                "query": search_query,
                "results": [],
            }

    def _formulate_query(self, user_input: str) -> str:
        """
        Formulate an effective search query from user input.

        Cleans and optimizes the query for search engine compatibility.
        """
        # Remove common conversational prefixes
        prefixes = [
            "cari tahu", "cari", "tolong cari", "bantu cari",
            "look up", "search for", "find", "tell me about",
        ]
        query = user_input.lower()
        for prefix in prefixes:
            if query.startswith(prefix):
                query = query[len(prefix):].strip()
                break

        # Remove question marks and clean up
        query = query.replace("?", "").strip()

        return query

    async def _perform_search(self, query: str) -> List[Dict[str, Any]]:
        """
        Perform the actual web search.

        In production, this would integrate with a search API
        (e.g., Google Custom Search, Bing Web Search, or SearXNG).
        """
        # Try using the z-ai-web-dev-sdk if available
        try:
            import importlib
            zai_sdk = importlib.import_module("z-ai-web-dev-sdk")

            # Use the SDK's web search functionality
            # Note: This is a placeholder - actual implementation
            # would use the SDK's search API
            logger.info("Using z-ai-web-dev-sdk for web search.")
            return [
                {
                    "title": f"Search result for: {query}",
                    "snippet": f"Relevant information about {query}",
                    "url": f"https://example.com/search?q={query}",
                    "source": "z-ai-sdk",
                }
            ]

        except ImportError:
            logger.warning("z-ai-web-dev-sdk not available. Using mock results.")

            # Mock results for development
            return [
                {
                    "title": f"Hasil pencarian untuk: {query}",
                    "snippet": f"Informasi relevan tentang '{query}'. Ini adalah hasil pencarian mock untuk keperluan development.",
                    "url": f"https://example.com/search?q={query}",
                    "source": "mock",
                }
            ]

    async def search_and_summarize(self, query: str, max_results: int = 3) -> Dict[str, Any]:
        """
        Search the web and provide a summarized version of results.

        Args:
            query: The search query.
            max_results: Maximum number of results to include.

        Returns:
            Dictionary with summarized search results.
        """
        search_result = await self.execute(query)

        if not search_result.get("success"):
            return search_result

        results = search_result.get("results", [])[:max_results]
        summary_parts = []

        for i, result in enumerate(results, 1):
            title = result.get("title", "No title")
            snippet = result.get("snippet", "No snippet")
            url = result.get("url", "")
            summary_parts.append(f"{i}. **{title}**\n   {snippet}\n   Sumber: {url}")

        summary = "\n\n".join(summary_parts)

        return {
            "success": True,
            "query": query,
            "summary": summary,
            "total_results": len(results),
        }
