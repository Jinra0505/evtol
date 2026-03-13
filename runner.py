import sys
from project.src.runner import main

if __name__ == "__main__":
    argv = [sys.argv[0]]
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        argv.extend(["--data", sys.argv[1]])
        argv.extend(sys.argv[2:])
    else:
        argv.extend(sys.argv[1:])
    sys.argv = argv
    main()
