import os
import sys
import subprocess

# from source base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# pip base directory
if not os.path.exists(os.path.join(BASE_DIR,'biomedisa_features')):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # add BASE_DIR to PYTHONPATH for absolute imports
    sys.path.append(BASE_DIR)

    # metadata
    import importlib_metadata
    metadata = importlib_metadata.metadata("biomedisa")

    __all__ = (
        "__title__",
        "__summary__",
        "__url__",
        "__version__",
        "__author__",
        "__email__",
        "__license__",
        "__copyright__",
    )

    __copyright__ = "Copyright (c) 2019-2024 Philipp Lösel"
    __title__ = metadata["name"]
    __summary__ = metadata["summary"]
    __url__ = "https://biomedisa.info"
    __version__ = metadata["version"]
    __author__ = "Philipp Lösel"
    __email__ = metadata["author-email"]
    __license__ = "European Union Public Licence 1.2 (EUPL 1.2)"

# biomedisa version when installed from source
else:
    try:
        if os.path.exists(os.path.join(BASE_DIR,'.git')):
            __version__ = subprocess.check_output(['git', 'describe', '--tags', '--always'], cwd=BASE_DIR).decode('utf-8').strip()
            f = open(os.path.join(BASE_DIR,'log/biomedisa_version'), 'w')
            f.write(__version__)
            f.close()
        else:
            raise Exception()
    except:
        if os.path.isfile(os.path.join(BASE_DIR,'log/biomedisa_version')):
            __version__ = open(os.path.join(BASE_DIR,'log/biomedisa_version'), 'r').readline().rstrip('\n')
        else:
            __version__ = None

