"""
Automated Endpoint (predict_model.py) Test for Sentiment Inference API
=======================================================================

This test script validates that the FastAPI server (predict_model)
is running correctly and returns valid JSON responses for sample inputs.

Usage:
Ensure FastAPI server is running:
    uv run python -m app.predict_model
In a new terminal, run the test client:
    uv run python -m app.test_inference

Response:
    Input: I love this video! It was super helpful and well explained.
    {
        "predictions": ["Positive"],
        "encoded_labels": [2],
        "probabilities": [[0.01, 0.02, 0.97]],
        "feature_shape": [1, 1004]
    }
"""

import requests
import json
import sys

API_URL = "http://127.0.0.1:8000/predict"

# Sample payloads for testing
TEST_PAYLOADS = [
    {
        "texts": ["I love this video! It was super helpful and well explained."]
    },  # Positive
    {"texts": ["This is terrible, I absolutely hate it."]},  # Negative
    {"texts": ["It’s okay, nothing special but not bad either."]},  # Neutral
]


def run_inference_test():
    """Sends test requests to the running FastAPI model server."""
    print("\n🚀 Running inference API tests...\n")

    for payload in TEST_PAYLOADS:
        try:
            response = requests.post(API_URL, json=payload, timeout=10)
            print(f"Input: {payload['texts'][0]}")

            if response.status_code == 200:
                result = response.json()
                print(json.dumps(result, indent=4))
            else:
                print(
                    f"❌ Request failed | Status {response.status_code} | {response.text}"
                )

        except requests.exceptions.RequestException as e:
            print(f"⚠️ Connection error: {e}")
            sys.exit(1)

        print("-" * 80)


if __name__ == "__main__":
    print("🧪 Starting FastAPI inference endpoint validation...\n")
    run_inference_test()
    print("\n✅ Inference test completed.\n")
