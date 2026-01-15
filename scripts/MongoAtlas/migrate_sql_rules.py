#!/usr/bin/env python3
"""
SQL Rules Migration Script

Migrates sql_rules.json to MongoDB with proper schema transformation.

Usage:
    python scripts/migrate_sql_rules.py [--dry-run] [--verbose] [--clear]

Options:
    --dry-run     Show what would be migrated without inserting
    --verbose     Enable verbose output
    --clear       Clear existing sql_rules collection before migration

Example:
    python scripts/migrate_sql_rules.py
    python scripts/migrate_sql_rules.py --dry-run
    python scripts/migrate_sql_rules.py --clear --verbose
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "python_services"))

try:
    from pymongo import MongoClient, UpdateOne
    from pymongo.errors import BulkWriteError, DuplicateKeyError
    from bson import ObjectId
except ImportError:
    print("ERROR: pymongo is required. Install with: pip install pymongo")
    sys.exit(1)

# Configuration
SQL_RULES_FILE = Path(__file__).parent.parent / "config" / "sql_rules.json"
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/?directConnection=true")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "rag_server")
COLLECTION_NAME = "sql_rules"

# Priority mapping
PRIORITY_MAP = {
    "critical": 100,
    "high": 75,
    "medium": 50,
    "low": 25
}


class SQLRulesMigrator:
    """Migrates SQL rules from JSON to MongoDB."""

    def __init__(self, mongodb_uri: str, database: str, verbose: bool = False, dry_run: bool = False):
        self.mongodb_uri = mongodb_uri
        self.database_name = database
        self.verbose = verbose
        self.dry_run = dry_run
        self.client = None
        self.db = None
        self.stats = {
            "global_rules": 0,
            "database_rules": 0,
            "total_rules": 0,
            "upserted": 0,
            "modified": 0,
            "errors": []
        }

    def connect(self) -> bool:
        """Connect to MongoDB."""
        if self.dry_run:
            self.log("DRY RUN: Would connect to MongoDB")
            return True

        try:
            # Connect without explicit write concern - use MongoDB defaults
            self.client = MongoClient(self.mongodb_uri)
            self.db = self.client[self.database_name]
            # Test connection
            self.client.admin.command('ping')
            self.log(f"Connected to MongoDB: {self.database_name}")
            return True
        except Exception as e:
            self.log(f"ERROR: Failed to connect to MongoDB: {e}", "ERROR")
            return False

    def disconnect(self):
        """Disconnect from MongoDB."""
        if self.client:
            self.client.close()
            self.log("Disconnected from MongoDB")

    def log(self, message: str, level: str = "INFO"):
        """Log a message."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prefix = f"[{timestamp}] [{level}]"

        if level == "DEBUG" and not self.verbose:
            return

        print(f"{prefix} {message}")

    def load_sql_rules(self) -> Optional[Dict]:
        """Load sql_rules.json file."""
        if not SQL_RULES_FILE.exists():
            self.log(f"ERROR: SQL rules file not found at {SQL_RULES_FILE}", "ERROR")
            return None

        try:
            with open(SQL_RULES_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.log(f"Loaded SQL rules from {SQL_RULES_FILE}")
            return data
        except Exception as e:
            self.log(f"ERROR: Failed to load SQL rules: {e}", "ERROR")
            return None

    def transform_global_rule(self, rule: Dict) -> Dict:
        """Transform a global constraint rule to MongoDB schema."""
        rule_id = rule.get("id", "")

        # Extract priority
        priority_str = rule.get("priority", "medium")
        if isinstance(priority_str, str):
            priority = PRIORITY_MAP.get(priority_str.lower(), 50)
        else:
            priority = 50

        # Build the MongoDB document
        doc = {
            "database": "_global",
            "rule_id": rule_id,
            "description": rule.get("description", ""),
            "type": rule.get("type", "constraint"),
            "trigger_keywords": [],  # Global rules don't have specific keywords
            "trigger_tables": [],
            "trigger_columns": [],
            "rule_text": rule.get("rule_text", ""),
            "priority": priority,
            "enabled": True,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }

        # Add auto_fix if present
        if "auto_fix" in rule:
            auto_fix = rule["auto_fix"]
            doc["auto_fix"] = {
                "pattern": auto_fix.get("pattern", ""),
                "replacement": auto_fix.get("replacement", "")
            }
            if "note" in auto_fix:
                doc["auto_fix"]["note"] = auto_fix["note"]
            if "applies_to_table" in auto_fix:
                doc["auto_fix"]["applies_to_table"] = auto_fix["applies_to_table"]

        return doc

    def transform_database_rule(self, rule: Dict, database: str) -> Dict:
        """Transform a database-specific rule to MongoDB schema."""
        rule_id = rule.get("id", "")

        # Extract priority
        priority_str = rule.get("priority", "medium")
        if isinstance(priority_str, str):
            priority = PRIORITY_MAP.get(priority_str.lower(), 50)
        else:
            priority = 50

        # Build the MongoDB document
        doc = {
            "database": database,
            "rule_id": rule_id,
            "description": rule.get("description", ""),
            "type": rule.get("type", "assistance"),
            "trigger_keywords": rule.get("trigger_keywords", []),
            "trigger_tables": rule.get("trigger_tables", []),
            "trigger_columns": rule.get("trigger_columns", []),
            "rule_text": rule.get("rule_text", ""),
            "priority": priority,
            "enabled": True,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }

        # Add auto_fix if present
        if "auto_fix" in rule:
            auto_fix = rule["auto_fix"]
            doc["auto_fix"] = {
                "pattern": auto_fix.get("pattern", ""),
                "replacement": auto_fix.get("replacement", "")
            }
            if "note" in auto_fix:
                doc["auto_fix"]["note"] = auto_fix["note"]
            if "applies_to_table" in auto_fix:
                doc["auto_fix"]["applies_to_table"] = auto_fix["applies_to_table"]

        # Add example if present
        if "example" in rule:
            example = rule["example"]
            doc["example"] = {
                "question": example.get("question", ""),
                "sql": example.get("sql", "")
            }

        return doc

    def migrate_rules(self, sql_rules_data: Dict) -> bool:
        """Migrate all rules to MongoDB."""
        operations = []

        # Process global constraints
        global_constraints = sql_rules_data.get("global_constraints", [])
        self.log(f"Processing {len(global_constraints)} global constraints...")

        for rule in global_constraints:
            try:
                doc = self.transform_global_rule(rule)
                self.stats["global_rules"] += 1

                if self.verbose:
                    self.log(f"  - Global rule: {doc['rule_id']}", "DEBUG")

                if not self.dry_run:
                    # Use upsert based on database + rule_id
                    filter_doc = {
                        "database": doc["database"],
                        "rule_id": doc["rule_id"]
                    }
                    # Update updated_at on update, but preserve created_at
                    update_doc = doc.copy()
                    update_doc.pop("created_at", None)

                    operations.append(UpdateOne(
                        filter_doc,
                        {
                            "$set": update_doc,
                            "$setOnInsert": {"created_at": datetime.utcnow()}
                        },
                        upsert=True
                    ))
            except Exception as e:
                self.log(f"ERROR: Failed to transform global rule {rule.get('id')}: {e}", "ERROR")
                self.stats["errors"].append(f"Global rule {rule.get('id')}: {e}")

        # Process database-specific rules
        database_rules = sql_rules_data.get("database_rules", {})
        self.log(f"Processing rules for {len(database_rules)} databases...")

        for database, db_config in database_rules.items():
            rules = db_config.get("rules", [])
            self.log(f"  - {database}: {len(rules)} rules")

            for rule in rules:
                try:
                    doc = self.transform_database_rule(rule, database)
                    self.stats["database_rules"] += 1

                    if self.verbose:
                        self.log(f"    - Rule: {doc['rule_id']}", "DEBUG")

                    if not self.dry_run:
                        # Use upsert based on database + rule_id
                        filter_doc = {
                            "database": doc["database"],
                            "rule_id": doc["rule_id"]
                        }
                        # Update updated_at on update, but preserve created_at
                        update_doc = doc.copy()
                        update_doc.pop("created_at", None)

                        operations.append(UpdateOne(
                            filter_doc,
                            {
                                "$set": update_doc,
                                "$setOnInsert": {"created_at": datetime.utcnow()}
                            },
                            upsert=True
                        ))
                except Exception as e:
                    self.log(f"ERROR: Failed to transform {database} rule {rule.get('id')}: {e}", "ERROR")
                    self.stats["errors"].append(f"{database} rule {rule.get('id')}: {e}")

        self.stats["total_rules"] = self.stats["global_rules"] + self.stats["database_rules"]

        # Execute writes if not dry run
        # Note: Using individual replace_one instead of bulk_write due to persistence issues
        if not self.dry_run and operations:
            try:
                self.log(f"\nInserting {len(operations)} rules into MongoDB...")
                collection = self.db[COLLECTION_NAME]

                inserted_count = 0
                updated_count = 0

                for operation in operations:
                    # Extract filter and document from UpdateOne operation
                    filter_doc = operation._filter
                    update_spec = operation._doc

                    # Get the document from $set
                    doc_to_upsert = update_spec.get('$set', {})

                    # Add created_at from $setOnInsert if this is a new document
                    if '$setOnInsert' in update_spec:
                        doc_to_upsert = doc_to_upsert.copy()
                        doc_to_upsert['created_at'] = update_spec['$setOnInsert']['created_at']

                    # Use replace_one with upsert
                    result = collection.replace_one(filter_doc, doc_to_upsert, upsert=True)

                    if result.upserted_id:
                        inserted_count += 1
                    elif result.modified_count > 0:
                        updated_count += 1

                self.stats["upserted"] = inserted_count
                self.stats["modified"] = updated_count
                self.log(f"Write complete: {inserted_count} inserted, {updated_count} updated")

                # Verify the write
                after_count = collection.count_documents({})
                self.log(f"Verification: {after_count} documents in collection")

            except Exception as e:
                self.log(f"ERROR: Write failed: {e}", "ERROR")
                import traceback
                self.log(traceback.format_exc(), "DEBUG")
                return False

        return True

    def create_indexes(self) -> bool:
        """Create indexes on the sql_rules collection."""
        if self.dry_run:
            self.log("DRY RUN: Would create indexes")
            return True

        try:
            collection = self.db[COLLECTION_NAME]

            # Create indexes
            self.log("Creating indexes...")

            # Index for querying by database and enabled status
            collection.create_index([("database", 1), ("enabled", 1)], name="database_enabled_idx")
            self.log("  - Created: database_enabled_idx")

            # Index for keyword matching
            collection.create_index([("trigger_keywords", 1)], name="trigger_keywords_idx")
            self.log("  - Created: trigger_keywords_idx")

            # Index for table matching
            collection.create_index([("trigger_tables", 1)], name="trigger_tables_idx")
            self.log("  - Created: trigger_tables_idx")

            # Unique index on database + rule_id
            collection.create_index(
                [("database", 1), ("rule_id", 1)],
                unique=True,
                name="database_rule_id_unique_idx"
            )
            self.log("  - Created: database_rule_id_unique_idx (unique)")

            return True
        except Exception as e:
            self.log(f"ERROR: Failed to create indexes: {e}", "ERROR")
            return False

    def clear_collection(self) -> bool:
        """Clear the sql_rules collection."""
        if self.dry_run:
            self.log("DRY RUN: Would clear sql_rules collection")
            return True

        try:
            result = self.db[COLLECTION_NAME].delete_many({})
            self.log(f"Cleared {result.deleted_count} existing rules from collection")
            return True
        except Exception as e:
            self.log(f"ERROR: Failed to clear collection: {e}", "ERROR")
            return False

    def print_summary(self):
        """Print migration summary."""
        self.log("\n" + "="*60)
        self.log("MIGRATION SUMMARY")
        self.log("="*60)
        self.log(f"Global rules processed: {self.stats['global_rules']}")
        self.log(f"Database rules processed: {self.stats['database_rules']}")
        self.log(f"Total rules: {self.stats['total_rules']}")

        if not self.dry_run:
            self.log(f"Rules inserted: {self.stats['upserted']}")
            self.log(f"Rules updated: {self.stats['modified']}")

        if self.stats["errors"]:
            self.log(f"\nErrors ({len(self.stats['errors'])}):", "ERROR")
            for error in self.stats["errors"][:10]:  # Show first 10 errors
                self.log(f"  - {error}", "ERROR")
            if len(self.stats["errors"]) > 10:
                self.log(f"  - ... and {len(self.stats['errors']) - 10} more errors", "ERROR")

    def run(self, clear_existing: bool = False) -> bool:
        """Run the migration."""
        self.log("="*60)
        self.log("SQL Rules Migration")
        self.log(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if self.dry_run:
            self.log("MODE: DRY RUN (no changes will be made)")
        self.log("="*60)

        # Load SQL rules
        sql_rules_data = self.load_sql_rules()
        if not sql_rules_data:
            return False

        # Connect to MongoDB
        if not self.connect():
            return False

        try:
            # Clear existing collection if requested
            if clear_existing:
                if not self.clear_collection():
                    return False

            # Migrate rules
            if not self.migrate_rules(sql_rules_data):
                return False

            # Create indexes
            if not self.create_indexes():
                self.log("WARNING: Index creation failed, but migration succeeded", "WARNING")

            # Final verification before disconnecting
            if not self.dry_run:
                # Force a sync with the database
                try:
                    self.client.admin.command('ping')
                except:
                    pass

                final_count = self.db[COLLECTION_NAME].count_documents({})
                self.log(f"\nFinal verification before disconnect: {final_count} documents in collection")

                # Additional verification: list first 3 rule_ids
                sample_docs = list(self.db[COLLECTION_NAME].find({}, {'_id': 0, 'rule_id': 1}).limit(3))
                if sample_docs:
                    self.log(f"Sample rules: {[d['rule_id'] for d in sample_docs]}")

            # Print summary
            self.print_summary()

            return len(self.stats["errors"]) == 0

        finally:
            self.disconnect()


def main():
    parser = argparse.ArgumentParser(
        description="Migrate SQL rules from JSON to MongoDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/migrate_sql_rules.py                    # Migrate rules (upsert mode)
    python scripts/migrate_sql_rules.py --dry-run          # Preview migration
    python scripts/migrate_sql_rules.py --clear            # Clear and re-import all rules
    python scripts/migrate_sql_rules.py --verbose          # Enable verbose output
        """
    )

    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be migrated without inserting"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    parser.add_argument(
        "--clear", "-c",
        action="store_true",
        help="Clear existing sql_rules collection before migration"
    )

    parser.add_argument(
        "--mongodb-uri",
        type=str,
        default=MONGODB_URI,
        help=f"MongoDB connection URI (default: {MONGODB_URI})"
    )

    parser.add_argument(
        "--database",
        type=str,
        default=MONGODB_DATABASE,
        help=f"MongoDB database name (default: {MONGODB_DATABASE})"
    )

    args = parser.parse_args()

    migrator = SQLRulesMigrator(
        mongodb_uri=args.mongodb_uri,
        database=args.database,
        verbose=args.verbose,
        dry_run=args.dry_run
    )

    success = migrator.run(clear_existing=args.clear)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
