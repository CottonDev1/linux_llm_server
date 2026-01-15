"""
Test script for RulesService

This script demonstrates basic usage of the RulesService.
Run from python_services directory:
    python -m sql_pipeline.services.test_rules_service
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sql_pipeline.services.rules_service import RulesService


async def test_rules_service():
    """Test basic RulesService functionality."""
    print("=" * 60)
    print("RulesService Test")
    print("=" * 60)

    # Get singleton instance
    print("\n1. Getting RulesService instance...")
    rules_service = await RulesService.get_instance()
    print("   Instance created successfully!")

    # Test get_rules
    print("\n2. Loading rules for 'EWRCentral' database...")
    try:
        rules = await rules_service.get_rules("EWRCentral", include_global=True)
        print(f"   Loaded {len(rules)} rules")
        if rules:
            print(f"   Example rule: {rules[0].rule_id}")
    except Exception as e:
        print(f"   Error loading rules: {e}")

    # Test exact match
    print("\n3. Testing exact match...")
    test_question = "Show tickets created today"
    try:
        match = await rules_service.find_exact_match(test_question, "EWRCentral")
        if match:
            print(f"   FOUND: Rule '{match.rule_id}'")
            print(f"   SQL: {match.example.sql[:80]}...")
        else:
            print(f"   No exact match for: {test_question}")
    except Exception as e:
        print(f"   Error: {e}")

    # Test similarity search
    print("\n4. Testing similarity search...")
    test_question = "tickets created status type"
    try:
        similar = await rules_service.find_similar_rules(test_question, "EWRCentral", threshold=0.3)
        print(f"   Found {len(similar)} similar rules")
        for rule, score in similar[:3]:
            print(f"   - {rule.rule_id} (similarity: {score:.2f})")
    except Exception as e:
        print(f"   Error: {e}")

    # Test keyword matching
    print("\n5. Testing keyword matching...")
    test_question = "show me ticket types and status"
    try:
        matches = await rules_service.find_keyword_matches(test_question, "EWRCentral")
        print(f"   Found {len(matches)} keyword matches")
        for rule in matches[:3]:
            print(f"   - {rule.rule_id}")
    except Exception as e:
        print(f"   Error: {e}")

    # Test auto-fixes
    print("\n6. Getting auto-fix patterns...")
    try:
        auto_fixes = await rules_service.get_auto_fixes("EWRCentral")
        print(f"   Found {len(auto_fixes)} auto-fix patterns")
        if auto_fixes:
            print(f"   Example: {auto_fixes[0].pattern[:50]}")
    except Exception as e:
        print(f"   Error: {e}")

    # Test table-based matching
    print("\n7. Testing table-based matching...")
    test_tables = ["CentralTickets", "CentralUsers"]
    try:
        table_rules = await rules_service.get_relevant_rules_for_tables(test_tables, "EWRCentral")
        print(f"   Found {len(table_rules)} rules for tables: {test_tables}")
        for rule in table_rules[:3]:
            print(f"   - {rule.rule_id}")
    except Exception as e:
        print(f"   Error: {e}")

    # Test cache invalidation
    print("\n8. Testing cache invalidation...")
    rules_service.invalidate_cache("EWRCentral")
    print("   Cache invalidated for EWRCentral")

    # Test cache after invalidation
    print("\n9. Reloading rules after cache invalidation...")
    try:
        rules = await rules_service.get_rules("EWRCentral", include_global=False)
        print(f"   Reloaded {len(rules)} rules (local only)")
    except Exception as e:
        print(f"   Error: {e}")

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)


if __name__ == "__main__":
    print("\nNOTE: This test requires:")
    print("  1. MongoDB running at mongodb://localhost:27017")
    print("  2. Database 'rag_server' with collection 'sql_rules'")
    print("  3. Rules populated in the collection")
    print("\nStarting tests...\n")

    try:
        asyncio.run(test_rules_service())
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
