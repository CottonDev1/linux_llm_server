"""
Database Name Parsing Utilities for Python Services

Handles parsing of database names for MongoDB lookup and master database identification.

DATABASE NAMING RULES FOR TROUBLESHOOTING COPIES:

Users work with master databases that get copied and renamed with additional identifiers.
This parser normalizes these names to find the correct schema in MongoDB.

IMPORTANT: Only these three EWR projects are normalized:
- EWR.Warehouse
- EWR.Gin
- EWR.Marketing

All other databases (including other EWR.* patterns) use their full name.

RULE 1: EWR Master Project Databases (non-Home)
For the three master projects, normalize to EWR.Project.Entity:
- EWR.Warehouse.84687 → lookup: "ewr.warehouse.entity"
- EWR.Warehouse.Chad:CustomerA → lookup: "ewr.warehouse.entity"
- EWR.Marketing.Test:Client123 → lookup: "ewr.marketing.entity"
- EWR.Gin.Demo:Site456 → lookup: "ewr.gin.entity"
- EWR.Gin.MikeT:70240-CITY2025-20251230_1404 → lookup: "ewr.gin.entity"
Pattern: If database matches EWR.(Warehouse|Gin|Marketing).*, use first 2 parts + ".Entity"

RULE 2: Home Databases (for the three master projects only)
If database name contains "Home" for a master project, use first 2 parts + ".Home":
- EWR.Gin.Chris:Adobe:Home → lookup: "ewr.gin.home"
- EWR.Warehouse.Test:ClientX:Home → lookup: "ewr.warehouse.home"
Pattern: If "Home" appears in name for master project, use [Part1].[Part2].Home

RULE 3: All Other Databases
For all other databases (including non-master EWR databases), use full name:
- CentralData → lookup: "centraldata"
- EWRCentral → lookup: "ewrcentral"
- EWR.SomeOtherProject.Entity → lookup: "ewr.someotherproject.entity" (full name!)
Pattern: Use full name (lowercase) for anything not matching the three master projects
"""

from typing import Optional, Dict
from dataclasses import dataclass

# The three master EWR projects that get normalized
MASTER_EWR_PROJECTS = ['WAREHOUSE', 'GIN', 'MARKETING']


def is_master_ewr_project(project_name: str) -> bool:
    """
    Checks if the given project name is one of the three master EWR projects
    that should be normalized (Warehouse, Gin, Marketing)

    Args:
        project_name: The project name (second part of EWR.Project.Entity)

    Returns:
        True if this is a master project that should be normalized
    """
    return project_name.upper() in MASTER_EWR_PROJECTS


def get_database_lookup_key(database_name: str) -> str:
    """
    Parses a database name and returns the lookup key for MongoDB queries

    IMPORTANT: Only EWR.Warehouse, EWR.Gin, and EWR.Marketing databases are normalized.
    All other databases use their full name for lookup.

    Args:
        database_name: Full database name

    Returns:
        Lookup key for MongoDB (lowercase)

    Examples:
        >>> get_database_lookup_key('EWR.Warehouse.Sales')
        'ewr.warehouse.entity'
        >>> get_database_lookup_key('EWR.Warehouse.84687')
        'ewr.warehouse.entity'
        >>> get_database_lookup_key('EWR.Warehouse.Home')
        'ewr.warehouse.home'
        >>> get_database_lookup_key('EWR.Warehouse.Chad:CustomerA')
        'ewr.warehouse.entity'
        >>> get_database_lookup_key('EWR.Gin.Chris:Adobe:Home')
        'ewr.gin.home'
        >>> get_database_lookup_key('EWR.Gin.MikeT:70240-CITY2025-20251230_1404')
        'ewr.gin.entity'
        >>> get_database_lookup_key('EWR.SomeOtherProject.Entity')
        'ewr.someotherproject.entity'
        >>> get_database_lookup_key('MyCustomDB')
        'mycustomdb'
        >>> get_database_lookup_key('EWRCentral')
        'ewrcentral'
    """
    if not database_name or not isinstance(database_name, str):
        raise ValueError('Database name must be a non-empty string')

    trimmed = database_name.strip()

    # Check if it's an EWR database (starts with "EWR.")
    if trimmed.startswith('EWR.'):
        # First, remove any colon-separated suffixes (troubleshooting identifiers)
        # Example: "EWR.Warehouse.Chad:CustomerA" → "EWR.Warehouse.Chad"
        without_suffix = trimmed.split(':')[0]
        parts = without_suffix.split('.')

        # EWR databases should have format: EWR.Project or EWR.Project.Entity
        if len(parts) >= 2:
            project_name = parts[1]

            # Only normalize if this is one of the three master projects
            if is_master_ewr_project(project_name):
                # RULE 2: If database contains "Home" anywhere (even in suffixes), return ewr.project.home
                if 'HOME' in trimmed.upper():
                    return f'{parts[0]}.{parts[1]}.Home'.lower()

                # RULE 1: For master project databases, use first two parts + Entity
                # This handles customer copies like EWR.Warehouse.84687 → ewr.warehouse.entity
                # Schema is stored in MongoDB as ewr.gin.entity, ewr.warehouse.entity, etc.
                return f'{parts[0]}.{parts[1]}.Entity'.lower()

            # Non-master EWR projects: use full name (lowercase)
            # Example: EWR.SomeOtherProject.Entity → ewr.someotherproject.entity
            return without_suffix.lower()

    # For non-EWR databases, return the full name (lowercase for consistent lookup)
    return trimmed.lower()


def is_master_database(database_name: str) -> bool:
    """
    Determines if a database name represents a master EWR database

    IMPORTANT: Only the three master projects (Warehouse, Gin, Marketing) have "master" databases.
    All other databases are considered their own masters.

    Master databases follow the pattern:
    - EWR.Project (exactly 2 parts, no entity suffix) OR
    - EWR.Project.Home (for Home databases)

    Args:
        database_name: Database name to check

    Returns:
        True if this is a master EWR database

    Examples:
        >>> is_master_database('EWR.Warehouse')
        True
        >>> is_master_database('EWR.Warehouse.Home')
        True
        >>> is_master_database('EWR.Gin.Chris:Adobe:Home')
        False
        >>> is_master_database('EWR.Warehouse.Sales')
        False
        >>> is_master_database('EWR.SomeOtherProject')
        False
        >>> is_master_database('MyCustomDB')
        False
    """
    if not database_name or not isinstance(database_name, str):
        return False

    trimmed = database_name.strip()

    # Must start with "EWR."
    if not trimmed.startswith('EWR.'):
        return False

    # Check if it has colons (troubleshooting suffix) - if so, it's not a master
    if ':' in trimmed:
        return False

    parts = trimmed.split('.')

    # Must have at least 2 parts
    if len(parts) < 2:
        return False

    project_name = parts[1]

    # Only the three master projects can have "master" databases
    if not is_master_ewr_project(project_name):
        return False

    # Master databases have exactly 2 parts: EWR.Project
    if len(parts) == 2:
        return True

    # OR exactly 3 parts with "Home" as the entity: EWR.Project.Home
    if len(parts) == 3 and parts[2].upper() == 'HOME':
        return True

    return False


def get_master_database_name(database_name: str) -> str:
    """
    Gets the master database name from any EWR database variant

    IMPORTANT: Only the three master projects (Warehouse, Gin, Marketing) are normalized.
    All other databases return their full name as the "master".

    For master projects:
    - "Home" databases, master is EWR.Project.Home
    - Other databases, master is EWR.Project

    For all other databases:
    - Returns the full name (lowercase)

    Args:
        database_name: Any database name

    Returns:
        Master database name (lowercase)

    Examples:
        >>> get_master_database_name('EWR.Warehouse.Sales')
        'ewr.warehouse.entity'
        >>> get_master_database_name('EWR.Warehouse.84687')
        'ewr.warehouse.entity'
        >>> get_master_database_name('EWR.Warehouse.Home')
        'ewr.warehouse.home'
        >>> get_master_database_name('EWR.Warehouse.Chad:CustomerA')
        'ewr.warehouse.entity'
        >>> get_master_database_name('EWR.Gin.Chris:Adobe:Home')
        'ewr.gin.home'
        >>> get_master_database_name('EWR.Gin.MikeT:70240-CITY2025-20251230_1404')
        'ewr.gin.entity'
        >>> get_master_database_name('EWR.SomeOtherProject.Entity')
        'ewr.someotherproject.entity'
        >>> get_master_database_name('MyCustomDB')
        'mycustomdb'
        >>> get_master_database_name('EWRCentral')
        'ewrcentral'
    """
    if not database_name or not isinstance(database_name, str):
        raise ValueError('Database name must be a non-empty string')

    trimmed = database_name.strip()

    # For EWR databases, extract master database name
    if trimmed.startswith('EWR.'):
        # First, remove any colon-separated suffixes (troubleshooting identifiers)
        without_suffix = trimmed.split(':')[0]
        parts = without_suffix.split('.')

        if len(parts) >= 2:
            project_name = parts[1]

            # Only normalize if this is one of the three master projects
            if is_master_ewr_project(project_name):
                # RULE 2: If database contains "Home" anywhere, master is ewr.project.home
                if 'HOME' in trimmed.upper():
                    return f'{parts[0]}.{parts[1]}.Home'.lower()

                # RULE 1: For master project databases, master is ewr.project.entity
                # Schema is stored in MongoDB as ewr.gin.entity, ewr.warehouse.entity, etc.
                return f'{parts[0]}.{parts[1]}.Entity'.lower()

            # Non-master EWR projects: use full name as "master" (lowercase)
            return without_suffix.lower()

    # For non-EWR databases, return as lowercase for consistent lookup
    return trimmed.lower()


def is_ewr_database(database_name: str) -> bool:
    """
    Determines if a database name represents an EWR database

    Args:
        database_name: Database name to check

    Returns:
        True if this is an EWR database

    Examples:
        >>> is_ewr_database('EWR.Warehouse.Production')
        True
        >>> is_ewr_database('MyCustomDB')
        False
    """
    if not database_name or not isinstance(database_name, str):
        return False

    return database_name.strip().startswith('EWR.')


def get_ewr_project_name(database_name: str) -> Optional[str]:
    """
    Extracts the project name from an EWR database

    Args:
        database_name: EWR database name

    Returns:
        Project name or None if not an EWR database

    Examples:
        >>> get_ewr_project_name('EWR.Warehouse.Sales')
        'Warehouse'
        >>> get_ewr_project_name('EWR.Gin.Home')
        'Gin'
        >>> get_ewr_project_name('MyCustomDB')
        None
    """
    if not database_name or not isinstance(database_name, str):
        return None

    trimmed = database_name.strip()

    if not trimmed.startswith('EWR.'):
        return None

    parts = trimmed.split('.')
    return parts[1] if len(parts) >= 2 else None


def get_ewr_entity_name(database_name: str) -> Optional[str]:
    """
    Extracts the entity name from an EWR database

    Args:
        database_name: EWR database name

    Returns:
        Entity name or None if not an EWR database or no entity

    Examples:
        >>> get_ewr_entity_name('EWR.Warehouse.Sales')
        'Sales'
        >>> get_ewr_entity_name('EWR.Gin.Home')
        'Home'
        >>> get_ewr_entity_name('EWR.Warehouse')
        None
        >>> get_ewr_entity_name('MyCustomDB')
        None
    """
    if not database_name or not isinstance(database_name, str):
        return None

    trimmed = database_name.strip()

    if not trimmed.startswith('EWR.'):
        return None

    parts = trimmed.split('.')
    return parts[2] if len(parts) >= 3 else None


@dataclass
class DatabaseInfo:
    """Parsed database information"""
    original_name: str
    lookup_key: str
    master_database: str
    is_ewr: bool
    is_master: bool
    is_home_database: bool
    project_name: Optional[str]
    entity_name: Optional[str]


def parse_database_name(database_name: str) -> DatabaseInfo:
    """
    Formats a database info object with lookup key and master database name

    Args:
        database_name: Database name to analyze

    Returns:
        DatabaseInfo object with all parsed information

    Examples:
        >>> info = parse_database_name('EWR.Warehouse.Sales')
        >>> info.lookup_key
        'ewr.warehouse'
        >>> info.is_master
        False

        >>> info = parse_database_name('EWR.Warehouse.Home')
        >>> info.lookup_key
        'ewr.warehouse.home'
        >>> info.is_master
        True
    """
    trimmed = database_name.strip()
    is_home_db = 'HOME' in trimmed.upper()

    return DatabaseInfo(
        original_name=trimmed,
        lookup_key=get_database_lookup_key(trimmed),
        master_database=get_master_database_name(trimmed),
        is_ewr=is_ewr_database(trimmed),
        is_master=is_master_database(trimmed),
        is_home_database=is_home_db,
        project_name=get_ewr_project_name(trimmed),
        entity_name=get_ewr_entity_name(trimmed)
    )


# For backwards compatibility with simple lowercase normalization
def normalize_database_name(database_name: str) -> str:
    """
    Convenience function that returns the lookup key for a database.
    This is the main function to use when querying MongoDB for schema/procedure data.

    Args:
        database_name: Any database name

    Returns:
        Normalized lookup key (lowercase, with EWR master project normalization applied)

    Examples:
        >>> normalize_database_name('EWR.Warehouse.84687')
        'ewr.warehouse.entity'
        >>> normalize_database_name('EWR.Gin.MikeT:70240-CITY2025-20251230_1404')
        'ewr.gin.entity'
        >>> normalize_database_name('EWR.Gin.Chris:Adobe:Home')
        'ewr.gin.home'
        >>> normalize_database_name('EWRCentral')
        'ewrcentral'
        >>> normalize_database_name('EWR.SomeOther.Entity')
        'ewr.someother.entity'
    """
    return get_database_lookup_key(database_name)


if __name__ == '__main__':
    # Test cases
    test_cases = [
        # Master project databases - should normalize to ewr.project.entity
        ('EWR.Warehouse', 'ewr.warehouse.entity'),
        ('EWR.Warehouse.Sales', 'ewr.warehouse.entity'),
        ('EWR.Warehouse.84687', 'ewr.warehouse.entity'),
        ('EWR.Warehouse.Chad:CustomerA', 'ewr.warehouse.entity'),
        ('EWR.Gin', 'ewr.gin.entity'),
        ('EWR.Gin.Production', 'ewr.gin.entity'),
        ('EWR.Gin.MikeT:70240-CITY2025-20251230_1404', 'ewr.gin.entity'),
        ('EWR.Marketing', 'ewr.marketing.entity'),
        ('EWR.Marketing.Test:Client123', 'ewr.marketing.entity'),

        # Home databases - should normalize to ewr.project.home
        ('EWR.Warehouse.Home', 'ewr.warehouse.home'),
        ('EWR.Gin.Chris:Adobe:Home', 'ewr.gin.home'),
        ('EWR.Gin.MikeT:70240-CITY2025-20251230_1404:Home', 'ewr.gin.home'),
        ('EWR.Marketing.Client:Home', 'ewr.marketing.home'),

        # Non-master EWR databases - should NOT normalize (use full name)
        ('EWR.SomeOther', 'ewr.someother'),
        ('EWR.SomeOther.Entity', 'ewr.someother.entity'),
        ('EWR.NewProject.Testing', 'ewr.newproject.testing'),

        # Non-EWR databases - should use full name
        ('EWRCentral', 'ewrcentral'),
        ('MyCustomDB', 'mycustomdb'),
        ('CentralData', 'centraldata'),
    ]

    print("Testing database name normalization:")
    print("=" * 60)

    all_passed = True
    for input_name, expected in test_cases:
        result = get_database_lookup_key(input_name)
        status = "PASS" if result == expected else "FAIL"
        if result != expected:
            all_passed = False
            print(f"{status} {input_name}")
            print(f"   Expected: {expected}")
            print(f"   Got:      {result}")
        else:
            print(f"{status} {input_name} -> {result}")

    print("=" * 60)
    print("All tests passed!" if all_passed else "Some tests failed!")
