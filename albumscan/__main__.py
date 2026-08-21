# -*- coding: utf-8 -*-
import sys

from .cli import main

if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    sys.exit(main())
