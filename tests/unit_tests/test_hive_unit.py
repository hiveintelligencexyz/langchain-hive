"""Unit tests for Hive Intelligence API."""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import requests
from langchain_core.tools import ToolException

from langchain_hive.hive import HiveSearch, HiveSearchAPIWrapper


class TestHiveSearchAPIWrapperUnit:
    """Unit tests for HiveSearchAPIWrapper."""

    def setup_method(self):
        """Set up test cases."""
        self.api_key = "test_api_key"
        self.wrapper = HiveSearchAPIWrapper(api_key=self.api_key)
        self.mock_response = {
            "content": "Bitcoin's current price is $60,000 USD.",
            "sources": [{"name": "Test Source", "url": "https://example.com"}],
        }

    @patch("requests.post")
    def test_process_query_formats_request_correctly(self, mock_post):
        """Test that process_query formats the request correctly."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.json.return_value = self.mock_response
        mock_post.return_value = mock_response

        # Call the method
        self.wrapper.process_query(
            prompt="test prompt",
            temperature=0.5,
            top_k=10,
            top_p=0.9,
            include_data_sources=False,
            wallet="0x123456789",
        )

        # Verify the request was made with correct parameters
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert kwargs["headers"]["api-key"] == self.api_key
        assert kwargs["headers"]["Content-Type"] == "application/json"
        
        # Verify data contains correct parameters
        request_data = json.loads(kwargs["data"])
        assert request_data["prompt"] == "test prompt"
        assert request_data["temperature"] == 0.5
        assert request_data["top_k"] == 10
        assert request_data["top_p"] == 0.9
        assert request_data["include_data_sources"] is False
        assert request_data["wallet"] == "0x123456789"

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.post")
    async def test_aprocess_query_formats_request_correctly(self, mock_post):
        """Test that aprocess_query formats the request correctly."""
        # Setup mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = self.mock_response
        mock_post.return_value.__aenter__.return_value = mock_response

        # Call the method
        await self.wrapper.aprocess_query(
            prompt="test prompt",
            temperature=0.5,
            top_k=10,
            top_p=0.9,
            include_data_sources=False,
            wallet="0x123456789",
        )

        # Verify the request was made with correct parameters
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert kwargs["headers"]["api-key"] == self.api_key
        assert kwargs["headers"]["Content-Type"] == "application/json"
        
        # Verify json contains correct parameters
        request_json = kwargs["json"]
        assert request_json["prompt"] == "test prompt"
        assert request_json["temperature"] == 0.5
        assert request_json["top_k"] == 10
        assert request_json["top_p"] == 0.9
        assert request_json["include_data_sources"] is False
        assert request_json["wallet"] == "0x123456789"


class TestHiveSearchUnit:
    """Unit tests for HiveSearch tool."""

    def setup_method(self):
        """Set up test cases."""
        self.api_key = "test_api_key"
        self.tool = HiveSearch(api_key=self.api_key)
        self.mock_response = {
            "content": "Bitcoin's current price is $60,000 USD.",
            "sources": [{"name": "Test Source", "url": "https://example.com"}],
        }

    def test_tool_has_required_attributes(self):
        """Test that the tool has all required attributes."""
        assert hasattr(self.tool, "name")
        assert hasattr(self.tool, "description")
        assert hasattr(self.tool, "args_schema")
        assert hasattr(self.tool, "handle_tool_error")
        assert hasattr(self.tool, "api_key")
        assert hasattr(self.tool, "base_url")
        assert hasattr(self.tool, "api_wrapper")

    @patch.object(HiveSearchAPIWrapper, "process_query")
    def test_run_with_prompt_only(self, mock_process_query):
        """Test that _run calls process_query with correct parameters."""
        mock_process_query.return_value = self.mock_response

        result = self.tool._run(prompt="test prompt")

        mock_process_query.assert_called_once_with(
            prompt="test prompt",
            messages=None,
            temperature=0.7,
            top_k=None,
            top_p=None,
            include_data_sources=True,
            wallet=None,
        )
        assert result == self.mock_response

    @pytest.mark.asyncio
    @patch.object(HiveSearchAPIWrapper, "aprocess_query")
    async def test_arun_with_prompt_only(self, mock_aprocess_query):
        """Test that _arun calls aprocess_query with correct parameters."""
        mock_aprocess_query.return_value = self.mock_response

        result = await self.tool._arun(prompt="test prompt")

        mock_aprocess_query.assert_called_once_with(
            prompt="test prompt",
            messages=None,
            temperature=0.7,
            top_k=None,
            top_p=None,
            include_data_sources=True,
            wallet=None,
        )
        assert result == self.mock_response 