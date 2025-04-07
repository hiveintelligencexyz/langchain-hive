"""Integration tests for Hive Intelligence API."""
import os
from unittest import mock

import pytest

from langchain_hive.hive import HiveSearch


@pytest.mark.skipif(
    "HIVE_INTELLIGENCE_API_KEY" not in os.environ,
    reason="HIVE_INTELLIGENCE_API_KEY environment variable not set",
)
class TestHiveIntegration:
    """Integration tests for Hive Intelligence API."""

    def setup_method(self):
        """Set up test cases."""
        self.api_key = os.environ["HIVE_INTELLIGENCE_API_KEY"]
        self.tool = HiveSearch(api_key=self.api_key)

    def test_search_with_prompt(self):
        """Test search with a simple prompt."""
        result = self.tool.invoke({"prompt": "What's the current price of Bitcoin?"})
        
        # Basic validation of the response
        assert isinstance(result, dict)
        # The actual response has fields like 'data_sources', 'fetchedData', 'response'
        # instead of 'content'
        assert "response" in result
        assert "data_sources" in result
        assert isinstance(result["data_sources"], list)
        assert len(result["data_sources"]) > 0

    def test_search_with_messages(self):
        """Test search with conversation history."""
        messages = [
            {"role": "user", "content": "Tell me about Uniswap"},
            {"role": "assistant", "content": "Uniswap is a decentralized exchange protocol..."},
            {"role": "user", "content": "What's its trading volume today?"},
        ]
        
        result = self.tool.invoke({"messages": messages})
        
        # Basic validation of the response - check for actual structure
        assert isinstance(result, dict)
        assert "response" in result
        assert "data_sources" in result
        assert isinstance(result["data_sources"], list)

    def test_search_with_parameters(self):
        """Test search with additional parameters."""
        result = self.tool.invoke({
            "prompt": "Explain the pros and cons of yield farming in DeFi",
            "temperature": 0.2,
            "top_p": 0.85,
            "top_k": 40,
            "include_data_sources": True
        })
        
        # Basic validation of the response - check for actual structure
        assert isinstance(result, dict)
        assert "response" in result
        assert "data_sources" in result
        assert isinstance(result["data_sources"], list) 