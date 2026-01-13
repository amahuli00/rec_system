#!/usr/bin/env python3
"""
Simple test client for the recommendation service API.

This script demonstrates how to interact with the service and test various scenarios.
"""

import requests
import json
import time
from typing import Dict, Any


class RecommendationClient:
    """Client for interacting with the recommendation service."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url

    def get_recommendations(self, user_id: int, k: int = 10) -> Dict[str, Any]:
        """Get recommendations for a user."""
        url = f"{self.base_url}/recommend"
        payload = {"user_id": user_id, "k": k}

        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response: {e.response.text}")
            return None

    def health_check(self) -> Dict[str, Any]:
        """Check service health."""
        url = f"{self.base_url}/health"
        try:
            response = requests.get(url, timeout=5)
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Health check failed: {e}")
            return None

    def get_info(self) -> Dict[str, Any]:
        """Get service information."""
        url = f"{self.base_url}/info"
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Info request failed: {e}")
            return None

    def readiness_check(self) -> Dict[str, Any]:
        """Check if service is ready."""
        url = f"{self.base_url}/ready"
        try:
            response = requests.get(url, timeout=5)
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Readiness check failed: {e}")
            return None


def main():
    """Run test scenarios."""
    client = RecommendationClient()

    print("=" * 70)
    print("Recommendation Service API Test Client")
    print("=" * 70)
    print()

    # Test 1: Root endpoint
    print("[Test 1] Root endpoint")
    try:
        response = requests.get(f"{client.base_url}/", timeout=5)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Failed: {e}")
    print()

    # Test 2: Health check
    print("[Test 2] Health check")
    health = client.health_check()
    if health:
        print(f"Status: {health.get('status')}")
        print(f"Checks: {json.dumps(health.get('checks', {}), indent=2)}")
    print()

    # Test 3: Readiness check
    print("[Test 3] Readiness check")
    ready = client.readiness_check()
    if ready:
        print(f"Response: {json.dumps(ready, indent=2)}")
    print()

    # Test 4: Service info
    print("[Test 4] Service information")
    info = client.get_info()
    if info:
        print(f"Service: {info.get('service_name')} v{info.get('version')}")
        print(f"Candidate Gen: {info['candidate_generation']['num_users']} users, "
              f"{info['candidate_generation']['num_items']} items")
        print(f"Ranking Model: {info['ranking']['model_name']}")
        print(f"Config: {json.dumps(info['config'], indent=2)}")
    print()

    # Test 5: Get recommendations
    print("[Test 5] Get recommendations for user 123")
    start_time = time.time()
    recs = client.get_recommendations(user_id=123, k=10)
    latency = (time.time() - start_time) * 1000

    if recs:
        print(f"User ID: {recs['user_id']}")
        print(f"Count: {recs['count']}")
        print(f"Service latency: {recs['latency_ms']:.1f}ms")
        print(f"End-to-end latency: {latency:.1f}ms")
        print(f"\nTop 5 recommendations:")
        for item in recs['recommendations'][:5]:
            print(f"  {item['rank']}. Movie {item['movie_id']}: "
                  f"score={item['score']:.3f}, "
                  f"similarity={item['candidate_similarity']:.3f}")
    print()

    # Test 6: Edge case - Large K
    print("[Test 6] Request large K (k=50)")
    recs = client.get_recommendations(user_id=123, k=50)
    if recs:
        print(f"Returned: {recs['count']} recommendations")
        print(f"Latency: {recs['latency_ms']:.1f}ms")
    print()

    # Test 7: Edge case - Cold start user
    print("[Test 7] Cold start user (user_id=999999)")
    recs = client.get_recommendations(user_id=999999, k=10)
    if recs:
        print(f"Returned: {recs['count']} recommendations")
        print(f"Latency: {recs['latency_ms']:.1f}ms")
    print()

    # Test 8: Invalid request
    print("[Test 8] Invalid request (k=0)")
    try:
        response = requests.post(
            f"{client.base_url}/recommend",
            json={"user_id": 123, "k": 0},
            timeout=5
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Failed: {e}")
    print()

    print("=" * 70)
    print("All tests completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
