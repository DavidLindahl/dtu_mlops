# tests/conftest.py
import sys
import os

# Get the path to the 'src' folder
# We go up one level from 'tests/' to root, then down into 'src'
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "..", "src")

# Add 'src' to the Python path for ALL tests
sys.path.append(src_path)
