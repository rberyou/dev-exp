#!/usr/bin/env python3
"""
Website Development Crew - AI-powered website development team

Usage:
    python main.py "Build a blog website with posts, comments, and user authentication"
    python main.py "Create an e-commerce site with product listings and shopping cart"
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent / '.env'
load_dotenv(env_path)

from crew import WebsiteDevCrew


def check_api_key():
    """Check if API key is configured."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        print("Error: OPENAI_API_KEY is not configured!")
        print("\nPlease do one of the following:")
        print("1. Copy .env.example to .env and add your API key:")
        print("   cp .env.example .env")
        print("   Then edit .env and set OPENAI_API_KEY=your_actual_key")
        print("\n2. Or set the environment variable directly:")
        print("   Windows: set OPENAI_API_KEY=your_actual_key")
        print("   Linux/Mac: export OPENAI_API_KEY=your_actual_key")
        sys.exit(1)
    return True


def run(requirements: str):
    """
    Run the website development crew with the given requirements.
    
    Args:
        requirements: User's requirements for the website
    """
    check_api_key()
    
    model = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")
    base_url = os.getenv("OPENAI_API_BASE", "default")
    
    print("=" * 60)
    print("Website Development Crew")
    print("=" * 60)
    print(f"\nModel: {model}")
    print(f"API Base: {base_url}")
    print(f"\nRequirements: {requirements}")
    print("-" * 60)
    print("Starting development process...\n")
    
    crew = WebsiteDevCrew()
    result = crew.crew().kickoff(inputs={"requirements": requirements})
    
    print("\n" + "=" * 60)
    print("Development Complete!")
    print("=" * 60)
    print(f"\nResult:\n{result}")
    
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    print(f"\nGenerated files are in: {output_dir}")
    
    return result


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py \"<your website requirements>\"")
        print("\nExample:")
        print('  python main.py "Build a blog with posts and comments"')
        print('  python main.py "Create a dashboard with charts and user profile"')
        sys.exit(1)
    
    requirements = sys.argv[1]
    run(requirements)


if __name__ == "__main__":
    main()
