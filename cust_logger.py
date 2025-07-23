import inspect
import logging
from colorama import Fore, Style, init
from datetime import datetime

init(autoreset=True)  # Automatically reset color formatting after each log
                      # Allowing different logs to have different colors

class ColorFormatter(logging.Formatter):
    """
    Custom logging formatter to add color and formatting to log messages.

    Attributes:
    -----------
    COLORS : dict
        Maps log levels to colorama Fore colors for log level emphasis.
    FILE_COLOR : str
        ANSI color and style for highlighting filename and line number.
    MESSAGE_COLOR_BY_FILE : dict
        Stores custom color for each file for consistent message coloring.

    Methods:
    --------
    format(record)
        Formats a log record with colored level, file info, and message.
    """
    COLORS = {
        logging.INFO: Fore.GREEN,
        logging.ERROR: Fore.RED,
        logging.WARNING: Fore.YELLOW
    }
    FILE_COLOR = Fore.CYAN + Style.BRIGHT
    MESSAGE_COLOR_BY_FILE = {}

    def format(self, record):
        """
        Formats the log record with colored level names, file location, and message content.

        Parameters:
        -----------
        record : logging.LogRecord
            The active log record to format.

        Returns:
        --------
        str
            The formatted and colorized log string.
        """
        log_color = self.COLORS.get(record.levelno, Style.RESET_ALL)
        levelname = f"{log_color}{record.levelname}{Style.RESET_ALL}"
        filename_lineno = f"{self.FILE_COLOR}{record.filename}:{record.lineno:<5}{Style.RESET_ALL}"
        message_color = self.MESSAGE_COLOR_BY_FILE.get(record.filename, Style.RESET_ALL)
        colored_message = f"{message_color}{record.msg}{Style.RESET_ALL}"
        record.timestamp = datetime.now().isoformat()
        log_output = f"{levelname}:     {filename_lineno} - {colored_message}"
        return log_output

color_formatter = ColorFormatter('%(levelname)s: %(filename_lineno)s - %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(color_formatter)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# Reduce logs from external libraries to warning or above to reduce clutter in console
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("fastapi").setLevel(logging.WARNING)

COLOR_MAP = {
    "BLACK": Fore.BLACK,
    "RED": Fore.RED,
    "GREEN": Fore.GREEN,
    "YELLOW": Fore.YELLOW,
    "PURPLE": Fore.BLUE,
    "MAGENTA": Fore.MAGENTA,
    "CYAN": Fore.CYAN,
    "WHITE": Fore.WHITE,
    "RESET": Style.RESET_ALL
}

def set_files_message_color(color_name):
    """
    Sets the display color for log messages from the calling file.

    Parameters:
    -----------
    color_name : str
        The name of the color to set for this file's log messages (case-insensitive).

    Effects:
    --------
    Updates the ColorFormatter's MESSAGE_COLOR_BY_FILE mapping and logs the update.
    Avoids redundant updates if color is already set.

    Error Handling:
    ---------------
    Logs a warning if ColorFormatter is not properly attached to the logger.
    """
    frame = inspect.stack()[1]
    caller_filename = frame.filename.split('/')[-1]
    color = COLOR_MAP.get(color_name.upper(), Style.RESET_ALL)

    if color_formatter:
        current_color = color_formatter.MESSAGE_COLOR_BY_FILE.get(caller_filename, None)
        if current_color == color:
            return
        color_formatter.MESSAGE_COLOR_BY_FILE[caller_filename] = color
        logger.info(f"Set message color for {caller_filename} to {color_name.upper()}")
    else:
        logger.warning(f"Could not find a ColorFormatter attached to the logger")
