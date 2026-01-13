import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pickle


def test_train_script_runs_and_creates_model():
    """Simple test that actually runs train.py script and verifies it creates a model file."""
    # 1. Get the absolute path to the project root (relative to THIS test file)
    # This works no matter where you run pytest from
    current_test_dir = Path(__file__).parent.resolve()
    project_root = current_test_dir.parent
    
    # 2. Construct the absolute path to train.py
    script_path = project_root / "src" / "cookiecutter_project" / "train.py"

    # Verify it exists before trying to run it (helps debugging)
    assert script_path.exists(), f"Could not find train.py at {script_path}"

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_svm_model.pkl"

        # 3. Run the script using the absolute path
        result = subprocess.run(
            ["uv", "run", "python", str(script_path), "train", "svm", "--output", str(output_path)],
            capture_output=True,
            text=True,
            cwd=project_root,  # IMPORTANT: Run the command as if we were inside the project folder
        )

        # Check that script ran successfully
        assert result.returncode == 0, (
            f"train.py script failed with return code {result.returncode}. "
            f"Error output: {result.stderr}. "
            f"Standard output: {result.stdout}"
        )

        # Check that model file was created
        assert output_path.exists(), (
            f"train.py should create model file at {output_path}, but file does not exist. "
            f"Script output: {result.stdout}"
        )

        # Check that file is not empty
        assert output_path.stat().st_size > 0, (
            f"Model file at {output_path} exists but is empty (size: {output_path.stat().st_size} bytes). "
            f"Model may not have been saved correctly."
        )

        # Verify the saved model can be loaded and used
        with open(output_path, "rb") as f:
            model = pickle.load(f)

        # Should be able to make predictions (breast cancer dataset has 30 features)
        test_data = np.random.rand(5, 30)
        predictions = model.predict(test_data)
        assert predictions.shape == (5,), (
            f"Model from train.py should predict shape (5,) for 5 samples, but got {predictions.shape}"
        )
        assert set(np.unique(predictions)).issubset({0, 1}), (
            f"Model from train.py should predict binary values (0 or 1), but got {set(np.unique(predictions))}"
        )
