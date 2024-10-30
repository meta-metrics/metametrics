import os
import random
import subprocess
import sys
from enum import Enum, unique

from metametrics.utils.logging import get_logger
from metametrics.tasks.run import run_metametrics

USAGE = (
    "=" * 70 + "\n"
    + "| Usage: metametrics-cli run -h: run metametrics                          |\n"
    + "=" * 70
)

WELCOME = (
    "=" * 92 + "\n" +
    "███╗   ███╗███████╗████████╗ █████╗ ███╗   ███╗███████╗████████╗██████╗ ██╗ ██████╗███████╗" + "\n"
    "████╗ ████║██╔════╝╚══██╔══╝██╔══██╗████╗ ████║██╔════╝╚══██╔══╝██╔══██╗██║██╔════╝██╔════╝" + "\n"
    "██╔████╔██║█████╗     ██║   ███████║██╔████╔██║█████╗     ██║   ██████╔╝██║██║     ███████╗" + "\n"
    "██║╚██╔╝██║██╔══╝     ██║   ██╔══██║██║╚██╔╝██║██╔══╝     ██║   ██╔══██╗██║██║     ╚════██║" + "\n"
    "██║ ╚═╝ ██║███████╗   ██║   ██║  ██║██║ ╚═╝ ██║███████╗   ██║   ██║  ██║██║╚██████╗███████║" + "\n"
    "╚═╝     ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝ ╚═════╝╚══════╝" + "\n"                                                                                
    "=" * 92 + "\n" +
    "| Welcome to MetaMetrics, version {}".format(VERSION).ljust(91) + "|\n" +
    "|" + " " * 90 + "|\n" +
    "| Project page: https://github.com/meta-metrics/metametrics".ljust(90) + "|\n"
    + "=" * 92
)

logger = get_logger(__name__)


@unique
class Command(str, Enum):
    RUN = "run"
    VER = "version"
    HELP = "help"

def main():
    command = sys.argv.pop(1) if len(sys.argv) != 1 else Command.HELP
    if command == Command.RUN:
        run_metametrics()
    elif command == Command.VER:
        print(WELCOME)
    elif command == Command.HELP:
        print(USAGE)
    else:
        raise NotImplementedError("Unknown command: {}.".format(command))
