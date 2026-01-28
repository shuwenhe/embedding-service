"""Simple tests for embedding_service."""

import unittest
from unittest.mock import Mock, patch
from embedding_service import clamp_provider


class TestConfig(unittest.TestCase):
    def test_clamp_provider(self):
        """Test provider normalization."""
        self.assertEqual(clamp_provider("ollama"), "ollama")
        self.assertEqual(clamp_provider("OLLAMA"), "ollama")
        self.assertEqual(clamp_provider("openai-compatible"), "openai-compatible")
        self.assertEqual(clamp_provider("invalid"), "ollama")
        self.assertEqual(clamp_provider(""), "ollama")
    
    @patch("langchain_ollama.OllamaEmbeddings")
    def test_build_embeddings_ollama(self, mock_ollama):
        """Test building Ollama embeddings."""
        from embedding_service import build_embeddings
        
        mock_instance = Mock()
        mock_ollama.return_value = mock_instance
        
        result = build_embeddings(
            provider="ollama",
            embed_model="test-model",
            base_url="http://localhost:11434"
        )
        
        mock_ollama.assert_called_once_with(
            model="test-model",
            base_url="http://localhost:11434"
        )
        self.assertEqual(result, mock_instance)


if __name__ == "__main__":
    unittest.main()
