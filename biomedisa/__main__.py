# biomedisa/__main__.py

import sys

def main():
    if len(sys.argv) < 2 or sys.argv[1] in ['-h','--help']:
        print("Usage: python3 -m biomedisa.<module_name> <args>")
        print("Modules available: interpolation, deeplearning, mesh")
        print("[-h, --help] for more information of each module")
        print("[-V, --version] for Biomedisa version installed")

    if sys.argv[1] in ['-v','-V','--version']:
        import biomedisa
        print(biomedisa.__version__)

if __name__ == "__main__":
    sys.exit(main())

