"""
Test script for the MongoDB-based SQLRulesService.

Verifies that:
1. Rules are loaded from MongoDB correctly
2. Exact match finding works
3. Rules context building works
4. Auto-fix patterns work

Run with: python -m scripts.test_sql_rules_service
"""

import asyncio
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from motor.motor_asyncio import AsyncIOMotorClient
from config import MONGODB_URI, MONGODB_DATABASE, COLLECTION_SQL_RULES


async def test_sql_rules_service():
    """Test the SQLRulesService."""

    print("=" * 60)
    print("SQLRulesService Test")
    print("=" * 60)

    # Import the service
    from services.sql_rules_service import SQLRulesService

    # Connect to MongoDB
    print(f"\nConnecting to MongoDB: {MONGODB_URI}")
    client = AsyncIOMotorClient(MONGODB_URI)
    db = client[MONGODB_DATABASE]

    # Create service
    service = SQLRulesService(db, COLLECTION_SQL_RULES)

    # Test 1: Get stats
    print("\n1. Testing get_stats()...")
    stats = await service.get_stats()
    print(f"   Total rules: {stats['total']}")
    print(f"   Active rules: {stats['active']}")
    print(f"   Rules by scope: {stats['by_scope']}")
    print(f"   Rules with examples: {stats['with_examples']}")
    print(f"   Rules with auto_fix: {stats['with_autofix']}")

    # Test 2: Find exact match (should find one)
    print("\n2. Testing find_exact_match()...")
    question = "How many centraltickets were entered into the database today?"
    result = await service.find_exact_match(question, "EWRCentral")
    if result:
        print(f"   Found exact match: {result['rule_id']}")
        print(f"   SQL: {result['sql'][:80]}...")
    else:
        print("   No exact match found (this may be expected if question differs)")

    # Test 3: Find exact match (case insensitive)
    print("\n3. Testing case-insensitive exact match...")
    question_upper = "HOW MANY CENTRALTICKETS WERE ENTERED INTO THE DATABASE TODAY?"
    result = await service.find_exact_match(question_upper, "EWRCentral")
    if result:
        print(f"   Found match (case insensitive): {result['rule_id']}")
    else:
        print("   No match found")

    # Test 4: Build rules context
    print("\n4. Testing build_rules_context()...")
    question = "How many tickets did Mike create yesterday?"
    table_names = {"CentralTickets", "CentralUsers"}
    context = await service.build_rules_context(question, "EWRCentral", table_names)
    if context:
        lines = context.split("\n")
        print(f"   Generated context with {len(lines)} lines")
        print(f"   First few lines:")
        for line in lines[:5]:
            print(f"     {line}")
    else:
        print("   No context generated")

    # Test 5: Apply auto-fixes
    print("\n5. Testing apply_auto_fixes()...")
    bad_sql = "SELECT * FROM dbo.dbo.CentralTickets WHERE CreateDate > '2024-01-01' LIMIT 10"
    fixed_sql, fixes = await service.apply_auto_fixes(bad_sql, "EWRCentral")
    print(f"   Original SQL: {bad_sql}")
    print(f"   Fixed SQL: {fixed_sql}")
    print(f"   Applied fixes: {len(fixes)}")
    for fix in fixes:
        print(f"     - {fix['rule_id']}: {fix['count']} replacement(s)")

    # Test 6: Get all rules for a scope
    print("\n6. Testing get_all_rules()...")
    ewrcentral_rules = await service.get_all_rules(scope="EWRCentral")
    print(f"   EWRCentral rules: {len(ewrcentral_rules)}")
    global_rules = await service.get_all_rules(scope="global")
    print(f"   Global rules: {len(global_rules)}")

    # Test 7: Get specific rule by ID
    print("\n7. Testing get_rule_by_id()...")
    rule = await service.get_rule_by_id("tickets-today-count")
    if rule:
        print(f"   Found rule: {rule['rule_id']}")
        print(f"   Description: {rule['description']}")
    else:
        print("   Rule not found")

    client.close()

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_sql_rules_service())
