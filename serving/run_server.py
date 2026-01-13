#!/usr/bin/env python3
"""
Simple script to run the recommendation service API server.

Usage:
    python serving/run_server.py

    Or with custom settings:
    python serving/run_server.py --host 0.0.0.0 --port 8080 --reload
"""

import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Run the recommendation service API")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on code changes (development only)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Logging level (default: info)"
    )

    args = parser.parse_args()

    print(f"Starting recommendation service on {args.host}:{args.port}")
    print(f"API docs available at: http://{args.host}:{args.port}/docs")
    print(f"Health check: http://{args.host}:{args.port}/health")

    uvicorn.run(
        "serving.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level
    )


if __name__ == "__main__":
    main()
