"""
Category Service - SQLite-based storage for document categories

This service provides a persistent storage layer for document categorization
using SQLite. Categories are organized in three tiers:
- Departments (Tier 1): Top-level organizational units
- Types (Tier 2): Document types within departments
- Subjects/Products (Tier 3): Optional granular categorization

The service uses aiosqlite for async database operations, maintaining
consistency with the project's async architecture pattern used in
mongodb_service.py and other services.

Database Location: ./data/categories.db (relative to project root)
"""
import os
import aiosqlite
from typing import Optional, List, Dict, Any
from datetime import datetime


# Database path - stored in the data directory alongside other project databases
# Uses relative path from this file's location to ../data/categories.db
DATABASE_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "data", "categories.db"
))

# Valid table names for security (prevents SQL injection via table name)
VALID_TABLES = {"departments", "types", "subjects_products"}

# Map category_type parameter to actual table names
# This allows using "subjects" as an alias for "subjects_products"
TABLE_MAP = {
    "departments": "departments",
    "types": "types",
    "subjects": "subjects_products",
    "subjects_products": "subjects_products"
}

# Default seed data for each category table
DEFAULT_DEPARTMENTS = [
    {"id": "customer_support", "name": "Customer Support", "description": "Customer service and support documentation", "display_order": 1},
    {"id": "development", "name": "Development", "description": "Software development and engineering documentation", "display_order": 2},
    {"id": "it", "name": "IT", "description": "Information technology and infrastructure documentation", "display_order": 3},
    {"id": "admin", "name": "Administration", "description": "Administrative and operational documentation", "display_order": 4},
    {"id": "general", "name": "General", "description": "General purpose documentation", "display_order": 5},
]

DEFAULT_TYPES = [
    {"id": "work_instruction", "name": "Work Instruction", "description": "Step-by-step procedural guides", "display_order": 1},
    {"id": "documentation", "name": "Documentation", "description": "Technical and reference documentation", "display_order": 2},
    {"id": "customer_information", "name": "Customer Information", "description": "Customer-facing information and guides", "display_order": 3},
    {"id": "procedural", "name": "Procedural", "description": "Process and workflow documentation", "display_order": 4},
]

DEFAULT_SUBJECTS = [
    {"id": "gin", "name": "Gin", "description": "Gin cotton processing system", "display_order": 1},
    {"id": "warehouse", "name": "Warehouse", "description": "Warehouse management system", "display_order": 2},
    {"id": "marketing", "name": "Marketing", "description": "Marketing and sales materials", "display_order": 3},
    {"id": "provider", "name": "Provider", "description": "Provider and vendor documentation", "display_order": 4},
]


class CategoryService:
    """
    Async SQLite service for managing document categories.

    Provides CRUD operations for three category tables:
    - departments: Top-level organizational categories
    - types: Document type classifications
    - subjects_products: Optional granular categorization

    Uses the singleton pattern with lazy initialization.
    The database and tables are created on first use.

    Example usage:
        service = CategoryService.get_instance()
        await service.initialize_db()

        # Get all departments
        departments = await service.get_all("departments")

        # Create a new type
        await service.create("types", "policy", "Policy", "Company policies", 5)
    """

    _instance: Optional['CategoryService'] = None

    def __init__(self, db_path: str = DATABASE_PATH):
        """
        Initialize the category service.

        Args:
            db_path: Path to the SQLite database file.
                    Defaults to data/categories.db in the project root.
        """
        self.db_path = db_path
        self._is_initialized = False

    @classmethod
    def get_instance(cls, db_path: str = DATABASE_PATH) -> 'CategoryService':
        """
        Get the singleton instance of CategoryService.

        Returns:
            The singleton CategoryService instance.
        """
        if cls._instance is None:
            cls._instance = cls(db_path)
        return cls._instance

    def _get_table_name(self, table_name: str) -> str:
        """
        Get the actual table name from a table_name or alias.

        Args:
            table_name: Table name or alias (e.g., 'subjects' or 'subjects_products')

        Returns:
            The actual table name in the database.

        Raises:
            ValueError: If table_name is not valid.
        """
        if table_name not in TABLE_MAP:
            raise ValueError(
                f"Invalid table name: '{table_name}'. "
                f"Must be one of: {', '.join(TABLE_MAP.keys())}"
            )
        return TABLE_MAP[table_name]

    async def initialize_db(self) -> Dict[str, Any]:
        """
        Initialize the database by creating tables if they don't exist.

        Creates the data directory if needed, then creates all three
        category tables with the standard schema including timestamps.
        If tables already exist, this operation is idempotent.

        Schema for each table:
            id TEXT PRIMARY KEY
            name TEXT NOT NULL
            description TEXT
            display_order INTEGER DEFAULT 0
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

        Returns:
            Dict with initialization status and details.
        """
        # Ensure the data directory exists
        data_dir = os.path.dirname(self.db_path)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)

        async with aiosqlite.connect(self.db_path) as db:
            # Create each category table with identical schema
            for table_name in VALID_TABLES:
                await db.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        description TEXT,
                        display_order INTEGER DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

            await db.commit()

        self._is_initialized = True

        return {
            "success": True,
            "message": "Database initialized successfully",
            "db_path": self.db_path,
            "tables": list(VALID_TABLES)
        }

    async def seed_default_data(self, force: bool = False) -> Dict[str, Any]:
        """
        Seed the database with default category values.

        Populates departments, types, and subjects_products tables
        with predefined default values. By default, only seeds tables
        that are empty to preserve existing data.

        Args:
            force: If True, delete existing data and reseed.
                   If False (default), only seed empty tables.

        Returns:
            Dict with seeding results for each table.
        """
        if not self._is_initialized:
            await self.initialize_db()

        results = {
            "departments": {"seeded": 0, "skipped": False},
            "types": {"seeded": 0, "skipped": False},
            "subjects_products": {"seeded": 0, "skipped": False},
        }

        async with aiosqlite.connect(self.db_path) as db:
            # Seed departments
            if force:
                await db.execute("DELETE FROM departments")

            cursor = await db.execute("SELECT COUNT(*) FROM departments")
            count = (await cursor.fetchone())[0]

            if count == 0 or force:
                for dept in DEFAULT_DEPARTMENTS:
                    await db.execute(
                        """
                        INSERT OR REPLACE INTO departments
                        (id, name, description, display_order)
                        VALUES (?, ?, ?, ?)
                        """,
                        (dept["id"], dept["name"], dept["description"], dept["display_order"])
                    )
                results["departments"]["seeded"] = len(DEFAULT_DEPARTMENTS)
            else:
                results["departments"]["skipped"] = True

            # Seed types
            if force:
                await db.execute("DELETE FROM types")

            cursor = await db.execute("SELECT COUNT(*) FROM types")
            count = (await cursor.fetchone())[0]

            if count == 0 or force:
                for doc_type in DEFAULT_TYPES:
                    await db.execute(
                        """
                        INSERT OR REPLACE INTO types
                        (id, name, description, display_order)
                        VALUES (?, ?, ?, ?)
                        """,
                        (doc_type["id"], doc_type["name"], doc_type["description"], doc_type["display_order"])
                    )
                results["types"]["seeded"] = len(DEFAULT_TYPES)
            else:
                results["types"]["skipped"] = True

            # Seed subjects_products
            if force:
                await db.execute("DELETE FROM subjects_products")

            cursor = await db.execute("SELECT COUNT(*) FROM subjects_products")
            count = (await cursor.fetchone())[0]

            if count == 0 or force:
                for subject in DEFAULT_SUBJECTS:
                    await db.execute(
                        """
                        INSERT OR REPLACE INTO subjects_products
                        (id, name, description, display_order)
                        VALUES (?, ?, ?, ?)
                        """,
                        (subject["id"], subject["name"], subject["description"], subject["display_order"])
                    )
                results["subjects_products"]["seeded"] = len(DEFAULT_SUBJECTS)
            else:
                results["subjects_products"]["skipped"] = True

            await db.commit()

        return {
            "success": True,
            "message": "Default data seeded successfully",
            "results": results
        }

    async def get_all(self, table_name: str) -> List[Dict[str, Any]]:
        """
        Get all categories from a table, ordered by display_order.

        Args:
            table_name: One of 'departments', 'types', 'subjects', or 'subjects_products'.

        Returns:
            List of category dictionaries with id, name, description,
            display_order, created_at, and updated_at fields.

        Raises:
            ValueError: If table_name is not valid.
        """
        actual_table = self._get_table_name(table_name)

        if not self._is_initialized:
            await self.initialize_db()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                f"SELECT * FROM {actual_table} ORDER BY display_order, name"
            )
            rows = await cursor.fetchall()

            return [dict(row) for row in rows]

    async def get_by_id(self, table_name: str, category_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a single category by its ID.

        Args:
            table_name: One of 'departments', 'types', 'subjects', or 'subjects_products'.
            category_id: The unique identifier of the category.

        Returns:
            Category dictionary if found, None otherwise.

        Raises:
            ValueError: If table_name is not valid.
        """
        actual_table = self._get_table_name(table_name)

        if not self._is_initialized:
            await self.initialize_db()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                f"SELECT * FROM {actual_table} WHERE id = ?",
                (category_id,)
            )
            row = await cursor.fetchone()

            return dict(row) if row else None

    async def create(
        self,
        table_name: str,
        category_id: str,
        name: str,
        description: Optional[str] = None,
        display_order: int = 0
    ) -> Dict[str, Any]:
        """
        Create a new category.

        Args:
            table_name: One of 'departments', 'types', 'subjects', or 'subjects_products'.
            category_id: Unique identifier for the category (e.g., 'it', 'hr').
            name: Display name for the category.
            description: Optional description of the category.
            display_order: Order for display (lower numbers appear first).

        Returns:
            Dict with success status and the created category.

        Raises:
            ValueError: If table_name is not valid or category already exists.
        """
        actual_table = self._get_table_name(table_name)

        if not self._is_initialized:
            await self.initialize_db()

        # Check if category already exists
        existing = await self.get_by_id(table_name, category_id)
        if existing:
            raise ValueError(f"Category with id '{category_id}' already exists in {actual_table}")

        now = datetime.utcnow().isoformat()

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                f"""
                INSERT INTO {actual_table}
                (id, name, description, display_order, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (category_id, name, description, display_order, now, now)
            )
            await db.commit()

        # Return the created category
        created = await self.get_by_id(table_name, category_id)

        return {
            "success": True,
            "message": f"Category '{category_id}' created in {actual_table}",
            "category": created
        }

    async def update(
        self,
        table_name: str,
        category_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        display_order: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Update an existing category.

        Only provided fields are updated; others remain unchanged.

        Args:
            table_name: One of 'departments', 'types', 'subjects', or 'subjects_products'.
            category_id: The ID of the category to update.
            name: New display name (optional).
            description: New description (optional).
            display_order: New display order (optional).

        Returns:
            Dict with success status and the updated category.

        Raises:
            ValueError: If table_name is not valid or category not found.
        """
        actual_table = self._get_table_name(table_name)

        if not self._is_initialized:
            await self.initialize_db()

        # Check if category exists
        existing = await self.get_by_id(table_name, category_id)
        if not existing:
            raise ValueError(f"Category with id '{category_id}' not found in {actual_table}")

        # Build update query dynamically based on provided fields
        updates = []
        params = []

        if name is not None:
            updates.append("name = ?")
            params.append(name)

        if description is not None:
            updates.append("description = ?")
            params.append(description)

        if display_order is not None:
            updates.append("display_order = ?")
            params.append(display_order)

        if not updates:
            return {
                "success": True,
                "message": "No fields to update",
                "category": existing
            }

        # Always update the updated_at timestamp
        updates.append("updated_at = ?")
        params.append(datetime.utcnow().isoformat())

        # Add the category_id for the WHERE clause
        params.append(category_id)

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                f"UPDATE {actual_table} SET {', '.join(updates)} WHERE id = ?",
                params
            )
            await db.commit()

        # Return the updated category
        updated = await self.get_by_id(table_name, category_id)

        return {
            "success": True,
            "message": f"Category '{category_id}' updated in {actual_table}",
            "category": updated
        }

    async def delete(self, table_name: str, category_id: str) -> Dict[str, Any]:
        """
        Delete a category.

        Args:
            table_name: One of 'departments', 'types', 'subjects', or 'subjects_products'.
            category_id: The ID of the category to delete.

        Returns:
            Dict with success status and message.

        Raises:
            ValueError: If table_name is not valid or category not found.
        """
        actual_table = self._get_table_name(table_name)

        if not self._is_initialized:
            await self.initialize_db()

        # Check if category exists
        existing = await self.get_by_id(table_name, category_id)
        if not existing:
            raise ValueError(f"Category with id '{category_id}' not found in {actual_table}")

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                f"DELETE FROM {actual_table} WHERE id = ?",
                (category_id,)
            )
            await db.commit()

        return {
            "success": True,
            "message": f"Category '{category_id}' deleted from {actual_table}"
        }

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about all category tables.

        Returns:
            Dict with count of categories in each table.
        """
        if not self._is_initialized:
            await self.initialize_db()

        stats = {}

        async with aiosqlite.connect(self.db_path) as db:
            for table_name in VALID_TABLES:
                cursor = await db.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = (await cursor.fetchone())[0]
                stats[table_name] = count

        return {
            "success": True,
            "db_path": self.db_path,
            "tables": stats,
            "total": sum(stats.values())
        }

    async def health_check(self) -> Dict[str, Any]:
        """
        Check the health of the category service.

        Returns:
            Dict with health status and database info.
        """
        try:
            if not self._is_initialized:
                await self.initialize_db()

            stats = await self.get_stats()

            return {
                "status": "healthy",
                "initialized": self._is_initialized,
                "db_path": self.db_path,
                "db_exists": os.path.exists(self.db_path),
                "tables": stats.get("tables", {})
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "initialized": self._is_initialized,
                "db_path": self.db_path,
                "db_exists": os.path.exists(self.db_path),
                "error": str(e)
            }


# Global instance getter for consistent access pattern with other services
def get_category_service() -> CategoryService:
    """
    Get the singleton CategoryService instance.

    This follows the same pattern as get_mongodb_service() and other
    service getters in the project for consistency.

    Returns:
        The singleton CategoryService instance.
    """
    return CategoryService.get_instance()


# Convenience function for initialization and seeding
async def initialize_categories(seed: bool = True) -> Dict[str, Any]:
    """
    Initialize the category database and optionally seed default data.

    Convenience function for bootstrapping the category system.
    Safe to call multiple times - only seeds empty tables.

    Args:
        seed: Whether to seed default data (default True).

    Returns:
        Dict with initialization and seeding results.
    """
    service = get_category_service()
    init_result = await service.initialize_db()

    if seed:
        seed_result = await service.seed_default_data()
        return {
            "initialization": init_result,
            "seeding": seed_result
        }

    return {
        "initialization": init_result,
        "seeding": {"skipped": True}
    }
