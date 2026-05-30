"""
AstraMind AI - Data Analyzer Tool
===================================
A tool for performing data analysis and statistical computations.
Supports descriptive statistics, trend analysis, and data summarization.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class DataAnalyzerTool:
    """
    A tool for data analysis and statistical computations.

    Supports:
    - Descriptive statistics (mean, median, mode, std dev)
    - Data summarization
    - Trend detection
    - Correlation analysis
    - Outlier detection
    """

    name = "data_analyzer"
    description = "Analyzes data and performs statistical computations."

    def __init__(self):
        self._data_cache: Dict[str, Any] = {}

    async def execute(self, query: str) -> Dict[str, Any]:
        """
        Execute data analysis based on the query.

        Args:
            query: A string describing the analysis to perform or containing data.

        Returns:
            Dictionary with analysis results.
        """
        logger.info(f"Data analyzer processing: {query}")

        # Determine the type of analysis requested
        analysis_type = self._determine_analysis_type(query)

        try:
            if analysis_type == "statistics":
                result = self._compute_statistics(query)
            elif analysis_type == "summary":
                result = self._generate_summary(query)
            elif analysis_type == "trend":
                result = self._analyze_trend(query)
            else:
                result = self._general_analysis(query)

            return {
                "success": True,
                "analysis_type": analysis_type,
                "result": result,
                "query": query,
            }

        except Exception as e:
            logger.error(f"Data analysis error: {e}")
            return {
                "success": False,
                "error": f"Kesalahan analisis data: {str(e)}",
                "analysis_type": analysis_type,
                "query": query,
            }

    def _determine_analysis_type(self, query: str) -> str:
        """Determine the type of analysis based on the query."""
        query_lower = query.lower()

        stat_keywords = ["statistik", "mean", "rata-rata", "median", "modus", "standar deviasi", "variansi", "statistical"]
        trend_keywords = ["tren", "trend", "pertumbuhan", "growth", "perubahan", "change", "naik", "turun"]
        summary_keywords = ["ringkasan", "summary", "rangkum", "ikhtisar", "overview"]

        for kw in stat_keywords:
            if kw in query_lower:
                return "statistics"
        for kw in trend_keywords:
            if kw in query_lower:
                return "trend"
        for kw in summary_keywords:
            if kw in query_lower:
                return "summary"

        return "general"

    def _compute_statistics(self, query: str) -> Dict[str, Any]:
        """Compute descriptive statistics for a dataset."""
        data = self._extract_numbers(query)

        if not data:
            return {"error": "Tidak dapat menemukan data numerik untuk dianalisis."}

        sorted_data = sorted(data)
        n = len(data)

        # Basic statistics
        mean = sum(data) / n
        median = sorted_data[n // 2] if n % 2 == 1 else (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2

        # Mode
        freq = {}
        for x in data:
            freq[x] = freq.get(x, 0) + 1
        max_freq = max(freq.values())
        mode = [k for k, v in freq.items() if v == max_freq]

        # Variance and standard deviation
        variance = sum((x - mean) ** 2 for x in data) / n
        std_dev = variance ** 0.5

        # Range
        data_range = sorted_data[-1] - sorted_data[0]

        # Quartiles
        q1_idx = n // 4
        q3_idx = 3 * n // 4
        q1 = sorted_data[q1_idx]
        q3 = sorted_data[q3_idx]
        iqr = q3 - q1

        # Outlier detection (IQR method)
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = [x for x in data if x < lower_bound or x > upper_bound]

        return {
            "count": n,
            "mean": round(mean, 4),
            "median": round(median, 4),
            "mode": mode if len(mode) < n else "No unique mode",
            "variance": round(variance, 4),
            "std_deviation": round(std_dev, 4),
            "min": sorted_data[0],
            "max": sorted_data[-1],
            "range": round(data_range, 4),
            "q1": round(q1, 4),
            "q3": round(q3, 4),
            "iqr": round(iqr, 4),
            "outliers": outliers,
            "outlier_count": len(outliers),
        }

    def _generate_summary(self, query: str) -> Dict[str, Any]:
        """Generate a data summary."""
        data = self._extract_numbers(query)

        if not data:
            return {"error": "Tidak dapat menemukan data untuk dirangkum."}

        stats = self._compute_statistics(query)

        summary_text = (
            f"Data terdiri dari {stats['count']} nilai. "
            f"Rata-rata: {stats['mean']}, median: {stats['median']}. "
            f"Rentang data dari {stats['min']} hingga {stats['max']} "
            f"(range: {stats['range']}). "
            f"Standar deviasi: {stats['std_deviation']}."
        )

        if stats["outlier_count"] > 0:
            summary_text += f" Terdeteksi {stats['outlier_count']} outlier."

        return {
            "summary": summary_text,
            "statistics": stats,
        }

    def _analyze_trend(self, query: str) -> Dict[str, Any]:
        """Analyze trends in the data."""
        data = self._extract_numbers(query)

        if not data or len(data) < 2:
            return {"error": "Data tidak cukup untuk analisis tren (minimal 2 titik data)."}

        # Simple linear trend
        n = len(data)
        x_vals = list(range(n))

        x_mean = sum(x_vals) / n
        y_mean = sum(data) / n

        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, data))
        denominator = sum((x - x_mean) ** 2 for x in x_vals)

        if denominator == 0:
            slope = 0
        else:
            slope = numerator / denominator

        intercept = y_mean - slope * x_mean

        # Trend direction
        if slope > 0.01:
            direction = "naik (meningkat)"
        elif slope < -0.01:
            direction = "turun (menurun)"
        else:
            direction = "stabil"

        # R-squared
        y_predicted = [slope * x + intercept for x in x_vals]
        ss_res = sum((y - yp) ** 2 for y, yp in zip(data, y_predicted))
        ss_tot = sum((y - y_mean) ** 2 for y in data)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        return {
            "trend_direction": direction,
            "slope": round(slope, 4),
            "intercept": round(intercept, 4),
            "r_squared": round(r_squared, 4),
            "data_points": n,
            "first_value": data[0],
            "last_value": data[-1],
            "change": round(data[-1] - data[0], 4),
            "change_percent": round(((data[-1] - data[0]) / data[0]) * 100, 2) if data[0] != 0 else None,
        }

    def _general_analysis(self, query: str) -> Dict[str, Any]:
        """Perform general data analysis."""
        data = self._extract_numbers(query)

        if not data:
            return {"error": "Tidak dapat menemukan data untuk dianalisis."}

        return {
            "data": data,
            "count": len(data),
            "basic_stats": self._compute_statistics(query),
        }

    def _extract_numbers(self, text: str) -> List[float]:
        """Extract numeric values from text."""
        import re
        numbers = re.findall(r'-?\d+\.?\d*', text)
        return [float(n) for n in numbers if n]
