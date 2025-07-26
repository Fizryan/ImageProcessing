# Logging_Config.py
# This module configures the logging settings for the application.

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s - (%(module)s:%(lineno)d)'
        },
    },
    
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
            'level': 'INFO',
            'stream': 'ext://sys.stdout',
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'formatter': 'standard',
            'filename': 'app.log',
            'maxBytes': 10485760,
            'backupCount': 5,
            'encoding': 'utf8',
            'level': 'DEBUG',
        },
    },
    
    'loggers': {
        '': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
        },
        'another_library': {
            'handlers': ['file'],
            'level': 'WARNING',
            'propagate': False,
        }
    }
}