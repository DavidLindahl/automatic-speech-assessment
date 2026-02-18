from pathlib import Path
import sys
import json
import unittest
from unittest.mock import patch
import asa.caption_generator as caption_generator

# Add src to path to import local modules
sys.path.append("src")


class TestCaptionGenerator(unittest.TestCase):
    @patch("asa.caption_generator.call_gemini_api")
    def test_process_mos_dataset(self, mock_api):
        """Test processing MOS dataset with mocked API."""
        print("\nTesting MOS dataset processing...")
        mock_api.return_value = "This is a mocked caption."

        input_path = "data/processed/mos_dataset.json"
        output_path = "data/processed/test_mos_predictions.json"

        # Ensure input exists
        if not Path(input_path).exists():
            print(f"Skipping MOS test: {input_path} not found.")
            return

        # Run processing on a subset (mocking json.load/dump would be cleaner but integration test is good too)
        # We'll just run it on the real file but limit calls by mocking/patching if needed,
        # or just let it run on all if file is small. User mentioned 10k items?
        # Let's mock json.load to only return 2 items to be fast.

        with open(input_path, "r") as f:
            full_data = json.load(f)
            sample_data = full_data[:2]

        # Create temp input
        temp_input = "data/processed/temp_test_mos.json"
        with open(temp_input, "w") as f:
            json.dump(sample_data, f)

        try:
            # Explicitly pass arguments to avoid default value being typer.Option object
            caption_generator.process_dataset_json(temp_input, output_path)

            # Check output
            with open(output_path, "r") as f:
                results = json.load(f)

            self.assertEqual(len(results), 2)
            self.assertIn("response", results[0])
            self.assertEqual(results[0]["response"], "This is a mocked caption.")
            # Check for query and flat metadata too if desired
            self.assertIn("query", results[0])
            self.assertIn("mos", results[0])
            print("MOS dataset processing verified.")

        finally:
            # Cleanup
            if Path(temp_input).exists():
                Path(temp_input).unlink()
            if Path(output_path).exists():
                Path(output_path).unlink()

    @patch("asa.caption_generator.call_gemini_api")
    def test_process_ab_dataset(self, mock_api):
        """Test processing A/B dataset with mocked API."""
        print("\nTesting A/B dataset processing...")

        # Mock API to return something that summarizes to "A"
        # First call is generation, second is summary.
        # But wait, summarize_ab_test calls call_gemini_api too.
        # So for 1 item, it calls API twice.

        # We can use side_effect for different returns
        def side_effect(prompt, **kwargs):
            if "SpeechA" in prompt or "SpeechB" in prompt:
                # This is likely the summary prompt or the AB prompt?
                # AB Prompt ends with "which speech is better:"
                # Summary prompt asks to "only output '[SpeechA]'..."

                if "Only output '[SpeechA]'" in prompt:
                    return "[SpeechA]"
                else:
                    return "Analysis: Speech A is better because..."
            return "Generic response"

        mock_api.side_effect = side_effect

        input_path = "data/processed/ab_dataset.json"
        output_path = "data/processed/test_ab_predictions.json"

        if not Path(input_path).exists():
            print(f"Skipping A/B test: {input_path} not found.")
            return

        with open(input_path, "r") as f:
            full_data = json.load(f)
            sample_data = full_data[:2]

        temp_input = "data/processed/temp_test_ab.json"
        with open(temp_input, "w") as f:
            json.dump(sample_data, f)

        try:
            # Explicitly pass arguments
            caption_generator.process_dataset_json(temp_input, output_path)

            with open(output_path, "r") as f:
                results = json.load(f)

            self.assertEqual(len(results), 2)
            self.assertIn("winner_predicted", results[0])
            # Our updated logic should strip brackets -> "A"
            self.assertEqual(results[0]["winner_predicted"], "A")
            self.assertIn("response", results[0])
            self.assertIn("query", results[0])
            # Check flattened metadata
            self.assertIn("A_mos", results[0])
            print("A/B dataset processing verified.")

        finally:
            if Path(temp_input).exists():
                Path(temp_input).unlink()
            if Path(output_path).exists():
                Path(output_path).unlink()


if __name__ == "__main__":
    unittest.main()
