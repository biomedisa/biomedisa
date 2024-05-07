# biomedisa/__main__.py

import os
import sys
import subprocess

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 -m biomedisa <module_name> <args>")
        sys.exit(1)
    
    module_name = sys.argv[1]
    args = sys.argv[2:]

    if module_name in ["interpolation", "deeplearning"]:
        cmd = ["python3", f"biomedisa_features/biomedisa_{module_name}.py"] + args
        subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    else:
        print("Unknown module:", module_name)

if __name__ == "__main__":
    sys.exit(main())

