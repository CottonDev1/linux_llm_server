"""
Configuration Parser Module

Unified parser for database configuration files supporting:
- JSON format (recommended)
- XML format (legacy, for backward compatibility)

Design Philosophy:
- Single responsibility: Parse configuration files into standardized format
- Format agnostic: Returns same structure regardless of input format
- Error resilient: Validates configurations and provides clear error messages

Usage:
    from sql_pipeline.extraction.config_parser import parse_config, DatabaseConfig

    # From file
    databases = await parse_config('./config.json')

    # From CLI args
    config = parse_cli_args(['--server', 'localhost', '--database', 'mydb', ...])
"""

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any


@dataclass
class DatabaseConfig:
    """Standard database configuration schema"""
    name: str
    server: str
    database: str
    lookup_key: str
    user: Optional[str] = None
    password: Optional[str] = None
    port: int = 1433
    options: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Normalize lookup_key to lowercase"""
        if self.lookup_key:
            self.lookup_key = self.lookup_key.lower()


async def parse_config(config_path: str) -> List[DatabaseConfig]:
    """
    Parse configuration file (auto-detects JSON or XML).

    Args:
        config_path: Path to configuration file (.json or .xml)

    Returns:
        List of DatabaseConfig objects

    Raises:
        ValueError: If file format is unsupported or content is invalid
        FileNotFoundError: If config file doesn't exist
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    print(f"Reading configuration from: {config_path}")

    content = path.read_text(encoding='utf-8')
    ext = path.suffix.lower()

    if ext == '.json':
        databases = _parse_json_config(content)
    elif ext == '.xml':
        databases = _parse_xml_config(content)
    else:
        raise ValueError(f"Unsupported config format: {ext}. Use .json or .xml")

    # Validate all configurations
    for i, db in enumerate(databases):
        _validate_database_config(db, i)

    print(f"Parsed {len(databases)} database configuration(s)")
    return databases


def _parse_json_config(json_content: str) -> List[DatabaseConfig]:
    """
    Parse JSON configuration file.

    Expected format:
    {
        "databases": [
            {
                "name": "gin",
                "server": "NCSQLTEST",
                "database": "EWR.Gin.Entity",
                "user": "EWRUser",
                "password": "xxx",
                "lookupKey": "gin"
            }
        ]
    }
    """
    try:
        config = json.loads(json_content)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parsing error: {e}")

    if not isinstance(config.get('databases'), list):
        raise ValueError('JSON config must have "databases" array at root level')

    databases = []
    for db in config['databases']:
        databases.append(DatabaseConfig(
            name=db.get('name') or db.get('lookupKey', 'unknown'),
            server=db.get('server', 'localhost'),
            database=db.get('database', ''),
            user=db.get('user'),
            password=db.get('password'),
            lookup_key=db.get('lookupKey') or db.get('name', 'unknown'),
            port=db.get('port', 1433),
            options=db.get('options', {})
        ))

    return databases


def _parse_xml_config(xml_content: str) -> List[DatabaseConfig]:
    """
    Parse XML configuration file (legacy format).

    Expected format:
    <DatabaseConfigurations>
        <Database>
            <PROJECT_CONTEXT>gin</PROJECT_CONTEXT>
            <NAME>EWR.Gin.Entity</NAME>
            <SERVER>NCSQLTEST</SERVER>
            <LOGIN_INFORMATION>
                <USER>EWRUser</USER>
                <PASSWORD>xxx</PASSWORD>
            </LOGIN_INFORMATION>
        </Database>
    </DatabaseConfigurations>
    """
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        raise ValueError(f"XML parsing error: {e}")

    databases = []

    # Find all Database elements
    for db_elem in root.findall('.//Database'):
        # Extract text content safely
        def get_text(elem_name: str, parent=None) -> Optional[str]:
            parent = parent if parent is not None else db_elem
            elem = parent.find(elem_name)
            return elem.text.strip() if elem is not None and elem.text else None

        # Get login information
        login_elem = db_elem.find('LOGIN_INFORMATION')
        user = get_text('USER', login_elem) if login_elem is not None else None
        password = get_text('PASSWORD', login_elem) if login_elem is not None else None

        # Parse port
        port_text = get_text('PORT')
        port = int(port_text) if port_text else 1433

        project_context = get_text('PROJECT_CONTEXT') or 'unknown'

        databases.append(DatabaseConfig(
            name=project_context,
            server=get_text('SERVER') or 'localhost',
            database=get_text('NAME') or '',
            user=user,
            password=password,
            lookup_key=project_context.lower(),
            port=port,
            options={}
        ))

    return databases


def _validate_database_config(config: DatabaseConfig, index: int):
    """
    Validate a database configuration.

    Args:
        config: Database configuration to validate
        index: Index in configuration array (for error messages)

    Raises:
        ValueError: If configuration is invalid
    """
    # Required fields - always require user and password for SQL auth
    required = ['server', 'database', 'lookup_key', 'user', 'password']

    missing = [field for field in required if not getattr(config, field)]

    if missing:
        raise ValueError(
            f"Database config at index {index} is missing required fields: {', '.join(missing)}"
        )

    # Validate server format (no SQL injection patterns)
    if ';' in config.server or '--' in config.server:
        raise ValueError(f"Database config at index {index}: Invalid server format")

    # Validate lookup key format (alphanumeric, lowercase, underscores)
    if not re.match(r'^[a-z0-9_]+$', config.lookup_key):
        raise ValueError(
            f"Database config at index {index}: lookup_key must be lowercase alphanumeric with underscores"
        )


def parse_cli_args(args: List[str]) -> DatabaseConfig:
    """
    Parse command-line arguments into database configuration.

    Supports:
        --server <server>
        --database <database>
        --user <user>
        --password <password>
        --lookup-key <key>
        --port <port>

    Args:
        args: List of command line arguments

    Returns:
        DatabaseConfig object

    Raises:
        ValueError: If required arguments are missing
    """
    config_dict = {
        'port': 1433,
        'options': {}
    }

    i = 0
    while i < len(args):
        arg = args[i]

        if not arg.startswith('--'):
            i += 1
            continue

        key = arg[2:].replace('-', '_')

        # Handle key-value arguments
        if i + 1 < len(args) and not args[i + 1].startswith('--'):
            value = args[i + 1]
            i += 2
        else:
            i += 1
            continue

        if key == 'server':
            config_dict['server'] = value
        elif key == 'database':
            config_dict['database'] = value
        elif key == 'user':
            config_dict['user'] = value
        elif key == 'password':
            config_dict['password'] = value
        elif key == 'lookup_key':
            config_dict['lookup_key'] = value
            config_dict['name'] = value
        elif key == 'port':
            config_dict['port'] = int(value)

    # Set defaults
    if 'name' not in config_dict:
        config_dict['name'] = config_dict.get('lookup_key', 'unknown')
    if 'lookup_key' not in config_dict:
        config_dict['lookup_key'] = config_dict.get('name', 'unknown')

    db_config = DatabaseConfig(**config_dict)
    _validate_database_config(db_config, 0)

    return db_config


def create_config_template() -> str:
    """
    Create a default JSON configuration template.
    Useful for initializing new projects.

    Returns:
        JSON template string
    """
    template = {
        "databases": [
            {
                "name": "example",
                "server": "localhost",
                "database": "ExampleDatabase",
                "user": "dbuser",
                "password": "your-password-here",
                "lookupKey": "example",
                "port": 1433,
                "options": {
                    "encrypt": True,
                    "trustServerCertificate": True
                }
            }
        ]
    }
    return json.dumps(template, indent=2)
