from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from core.config_util import get_config


def get_connect_string():
    server = get_config('DATABASE', 'server')
    database = get_config('DATABASE', 'database')
    driver = get_config('DATABASE', 'driver')

    return f"mssql://@{server}/{database}?driver={driver}"


class DbSession:
    """SQLAlchemy database connection"""

    def __init__(self):
        self.session = None
        self.engine = None

    def __enter__(self):
        self.engine = create_engine(get_connect_string())
        _session = sessionmaker()
        self.session = _session(bind=self.engine)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_value is None:
            self.session.commit()
            self.session.close()
        else:
            self.session.rollback()
            self.session.close()
            raise exc_value
        return True
