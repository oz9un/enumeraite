"""Test that the project structure is set up correctly."""
import sys
from pathlib import Path

def test_project_structure():
    """Test that all required directories and files exist."""
    project_root = Path(__file__).parent.parent

    # Test source directories
    assert (project_root / "src" / "enumeraite").exists()
    assert (project_root / "src" / "enumeraite" / "core").exists()
    assert (project_root / "src" / "enumeraite" / "providers").exists()
    assert (project_root / "src" / "enumeraite" / "cli").exists()

    # Test test directories
    assert (project_root / "tests").exists()
    assert (project_root / "tests" / "core").exists()
    assert (project_root / "tests" / "providers").exists()
    assert (project_root / "tests" / "cli").exists()

    # Test configuration files
    assert (project_root / "pyproject.toml").exists()
    assert (project_root / "requirements.txt").exists()

def test_package_imports():
    """Test that the package can be imported."""
    # Add src to path to allow imports
    project_root = Path(__file__).parent.parent
    src_path = project_root / "src"
    sys.path.insert(0, str(src_path))

    # Test basic import
    import enumeraite
    assert enumeraite.__version__ == "0.1.0"