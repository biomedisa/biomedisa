import subprocess
import os

if __name__ == "__main__":
    # Import config
    try:
        from config import python_path, lib_path
    except:
        from config_template import python_path, lib_path

    # Create a clean environment
    new_env = os.environ.copy()

    # Remove environment variables that may interfere
    for var in ["PYTHONHOME", "PYTHONPATH", "LD_LIBRARY_PATH"]:
        new_env.pop(var, None)

    # Set new pythonpath
    new_env["PYTHONPATH"] = lib_path

    # Print Python environment
    subprocess.run(
        [python_path, "-c", "import sys; print(sys.version); print(sys.executable); print(sys.path)"],
        env=new_env
    )

    # Show Biomedisa pip package
    subprocess.run(
        [python_path, "-m", "pip", "show", "biomedisa"],
        env=new_env
    )

    # Show Biomedisa installation (for example 26.4.2 for pip and v26.4.1-12-gdbfe877 for git)
    print("Biomedisa Version:")
    subprocess.run(
        [python_path, "-m", "biomedisa.interpolation", "-v"],
        env=new_env
    )

    # Show all available OpenCL platforms
    subprocess.run(
        [python_path, "-c", "import pyopencl as cl; print('Platforms:',cl.get_platforms())"],
        env=new_env
    )

    # Check if CUDA is working
    subprocess.run(
        [python_path, "-m", "biomedisa.features.pycuda_test"],
        env=new_env
    )

