# Load default configuration
import logging
import logging.config

# Constants
CONFIG_FILE = "cnf/logging.conf"

# Load config
logging.config.fileConfig(CONFIG_FILE)
