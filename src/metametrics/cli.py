import sys
from enum import Enum, unique

from metametrics.utils.logging import get_logger
from metametrics.utils.constants import VERSION

USAGE = (
    "\n" + "=" * 75 + "\n"
    + "| Usage: metametrics-cli run -h: run metametrics                          |\n"
    + "=" * 75
)

WELCOME = (
    "\n" + "=" * 92 + "\n" +
    "███╗   ███╗███████╗████████╗ █████╗ ███╗   ███╗███████╗████████╗██████╗ ██╗ ██████╗███████╗" + "\n"
    "████╗ ████║██╔════╝╚══██╔══╝██╔══██╗████╗ ████║██╔════╝╚══██╔══╝██╔══██╗██║██╔════╝██╔════╝" + "\n"
    "██╔████╔██║█████╗     ██║   ███████║██╔████╔██║█████╗     ██║   ██████╔╝██║██║     ███████╗" + "\n"
    "██║╚██╔╝██║██╔══╝     ██║   ██╔══██║██║╚██╔╝██║██╔══╝     ██║   ██╔══██╗██║██║     ╚════██║" + "\n"
    "██║ ╚═╝ ██║███████╗   ██║   ██║  ██║██║ ╚═╝ ██║███████╗   ██║   ██║  ██║██║╚██████╗███████║" + "\n"
    "╚═╝     ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝ ╚═════╝╚══════╝" + "\n"                                                                                
    "|" + "-" * 90 + "|" + "\n"
    + "| Welcome to MetaMetrics, version {}".format(VERSION).ljust(91) + "|\n" +
    "|" + " " * 90 + "|\n" +
    "| Project page: https://github.com/meta-metrics/metametrics".ljust(91) + "|\n"
    + "=" * 92
)

logger = get_logger(__name__)


@unique
class Command(str, Enum):
    RUN = "run"
    VER = "version"
    HELP = "help"

def main():
    logger.info(sys.argv)
    command = sys.argv.pop(1) if len(sys.argv) != 1 else Command.HELP
    if command == Command.RUN:
        from metametrics.tasks.run import run_metametrics
        run_metametrics()
    elif command == Command.VER:
        logger.info(WELCOME)
    elif command == Command.HELP:
        logger.info(USAGE)
    else:
        raise NotImplementedError("Unknown command: {}.".format(command))
