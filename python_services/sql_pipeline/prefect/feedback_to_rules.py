"""
Feedback to Rules Converter
===========================

Converts verified user corrections from agent_corrections collection
into SQL rules for the sql_rules collection.

This is the critical feedback loop that allows the system to learn
from user corrections automatically.
"""

import asyncio
import hashlib
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from prefect import flow, task, get_run_logger
from prefect.artifacts import create_markdown_artifact


@dataclass
class CorrectionPattern:
    """Pattern extracted from a correction."""
    pattern_type: str  # column_mapping, table_alias, join_pattern, etc.
    wrong_value: str
    correct_value: str
    tables: List[str] = field(default_factory=list)
    occurrences: int = 1
    examples: List[Dict] = field(default_factory=list)


@dataclass
class GeneratedRule:
    """Rule generated from patterns."""
    id: str
    database: str
    description: str
    type: str  # assistance, constraint
    trigger_keywords: List[str]
    trigger_tables: List[str]
    rule_text: str
    auto_fix: Optional[Dict] = None
    example: Optional[Dict] = None
    priority: str = "normal"
    source: str = "feedback_correction"


def extract_column_mapping(original_sql: str, corrected_sql: str) -> Optional[Tuple[str, str]]:
    """Extract column name changes between original and corrected SQL."""
    # Find columns that differ between queries
    orig_cols = set(re.findall(r'\b([A-Z][a-zA-Z0-9_]+)\b', original_sql))
    corr_cols = set(re.findall(r'\b([A-Z][a-zA-Z0-9_]+)\b', corrected_sql))

    removed = orig_cols - corr_cols
    added = corr_cols - orig_cols

    # Look for likely replacements (similar length, similar position)
    if len(removed) == 1 and len(added) == 1:
        wrong = list(removed)[0]
        correct = list(added)[0]
        # Verify this looks like a column name change (not table alias)
        if len(wrong) > 3 and len(correct) > 3:
            return (wrong, correct)

    return None


def extract_tables_from_sql(sql: str) -> List[str]:
    """Extract table names from SQL query."""
    # Match FROM and JOIN clauses
    tables = []

    # FROM clause
    from_match = re.findall(r'\bFROM\s+(\[?[a-zA-Z][a-zA-Z0-9_]*\]?)', sql, re.IGNORECASE)
    tables.extend([t.strip('[]') for t in from_match])

    # JOIN clauses
    join_match = re.findall(r'\bJOIN\s+(\[?[a-zA-Z][a-zA-Z0-9_]*\]?)', sql, re.IGNORECASE)
    tables.extend([t.strip('[]') for t in join_match])

    return list(set(tables))


def extract_keywords_from_question(question: str) -> List[str]:
    """Extract meaningful keywords from a question."""
    # Remove common words
    stop_words = {
        'show', 'me', 'the', 'all', 'a', 'an', 'of', 'in', 'on', 'at', 'to',
        'for', 'with', 'from', 'by', 'as', 'is', 'are', 'was', 'were', 'be',
        'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'could', 'should', 'may', 'might', 'must', 'can', 'get',
        'list', 'find', 'display', 'give', 'tell', 'what', 'which', 'where',
        'when', 'how', 'many', 'much', 'that', 'this', 'these', 'those'
    }

    words = re.findall(r'\b([a-zA-Z][a-zA-Z0-9_]*)\b', question.lower())
    keywords = [w for w in words if w not in stop_words and len(w) > 2]

    return keywords[:10]  # Limit to 10 keywords


@task(
    name="fetch_unprocessed_corrections",
    description="Fetch verified corrections not yet converted to rules",
    retries=2,
    tags=["feedback", "corrections"]
)
async def fetch_unprocessed_corrections_task(
    database: str,
    limit: int = 100
) -> List[Dict]:
    """Fetch corrections that haven't been converted to rules."""
    logger = get_run_logger()

    import sys
    sys.path.insert(0, '/mnt/c/Projects/llm_website-usingSQLFeedBack/python_services')

    from mongodb import get_mongodb_service

    mongo = await get_mongodb_service()

    # Find verified corrections that haven't been processed
    corrections = await mongo.db["agent_corrections"].find({
        "database": {"$regex": database, "$options": "i"},
        "verified": True,
        "converted_to_rule": {"$ne": True}
    }).sort("timestamp", -1).limit(limit).to_list(limit)

    logger.info(f"Found {len(corrections)} unprocessed corrections for {database}")
    return corrections


@task(
    name="analyze_correction",
    description="Analyze a correction for patterns",
    tags=["feedback", "analysis"]
)
async def analyze_correction_task(correction: Dict) -> Optional[CorrectionPattern]:
    """Analyze a single correction for extractable patterns."""
    logger = get_run_logger()

    original = correction.get("original_sql", "")
    corrected = correction.get("corrected_sql", "")
    question = correction.get("natural_language", "")
    reason = correction.get("correction_reason", "")

    if not original or not corrected:
        return None

    # Try to extract column mapping
    col_mapping = extract_column_mapping(original, corrected)
    if col_mapping:
        wrong_col, correct_col = col_mapping
        tables = extract_tables_from_sql(corrected)

        return CorrectionPattern(
            pattern_type="column_mapping",
            wrong_value=wrong_col,
            correct_value=correct_col,
            tables=tables,
            examples=[{
                "question": question,
                "original": original,
                "corrected": corrected,
                "reason": reason
            }]
        )

    # Check for table name corrections
    orig_tables = set(extract_tables_from_sql(original))
    corr_tables = set(extract_tables_from_sql(corrected))

    if orig_tables != corr_tables:
        removed = orig_tables - corr_tables
        added = corr_tables - orig_tables

        if len(removed) == 1 and len(added) == 1:
            wrong_table = list(removed)[0]
            correct_table = list(added)[0]

            return CorrectionPattern(
                pattern_type="table_mapping",
                wrong_value=wrong_table,
                correct_value=correct_table,
                tables=list(corr_tables),
                examples=[{
                    "question": question,
                    "original": original,
                    "corrected": corrected,
                    "reason": reason
                }]
            )

    return None


@task(
    name="group_patterns",
    description="Group similar patterns together",
    tags=["feedback", "grouping"]
)
async def group_patterns_task(
    patterns: List[CorrectionPattern]
) -> List[CorrectionPattern]:
    """Group similar patterns to identify frequent errors."""
    logger = get_run_logger()

    # Group by pattern key (type + wrong + correct)
    grouped = {}

    for pattern in patterns:
        if not pattern:
            continue

        key = f"{pattern.pattern_type}:{pattern.wrong_value}:{pattern.correct_value}"

        if key in grouped:
            grouped[key].occurrences += 1
            grouped[key].examples.extend(pattern.examples)
            grouped[key].tables = list(set(grouped[key].tables + pattern.tables))
        else:
            grouped[key] = pattern

    # Filter to patterns with 2+ occurrences or high confidence single corrections
    significant = [p for p in grouped.values() if p.occurrences >= 1]

    logger.info(f"Grouped into {len(significant)} significant patterns")
    return significant


@task(
    name="generate_rule_from_pattern",
    description="Generate SQL rule from pattern",
    tags=["feedback", "rules"]
)
async def generate_rule_from_pattern_task(
    pattern: CorrectionPattern,
    database: str
) -> Optional[GeneratedRule]:
    """Generate a SQL rule from an extracted pattern."""
    logger = get_run_logger()

    if pattern.pattern_type == "column_mapping":
        # Column name correction rule
        rule_id = f"feedback_{hashlib.md5(f'{pattern.wrong_value}_{pattern.correct_value}'.encode()).hexdigest()[:8]}"

        # Extract keywords from examples
        keywords = set()
        for ex in pattern.examples[:5]:
            keywords.update(extract_keywords_from_question(ex.get("question", "")))

        rule_text = f"Use '{pattern.correct_value}' instead of '{pattern.wrong_value}'. The column '{pattern.wrong_value}' does not exist."

        auto_fix = {
            "pattern": rf"\b{re.escape(pattern.wrong_value)}\b",
            "replacement": pattern.correct_value
        }

        # Get example from first occurrence
        example = None
        if pattern.examples:
            ex = pattern.examples[0]
            example = {
                "question": ex.get("question", ""),
                "sql": ex.get("corrected", "")
            }

        return GeneratedRule(
            id=rule_id,
            database=database,
            description=f"Column correction: {pattern.wrong_value} → {pattern.correct_value}",
            type="constraint",
            trigger_keywords=list(keywords)[:10],
            trigger_tables=pattern.tables,
            rule_text=rule_text,
            auto_fix=auto_fix,
            example=example,
            priority="high" if pattern.occurrences >= 3 else "normal",
            source="feedback_correction"
        )

    elif pattern.pattern_type == "table_mapping":
        # Table name correction rule
        rule_id = f"feedback_table_{hashlib.md5(f'{pattern.wrong_value}_{pattern.correct_value}'.encode()).hexdigest()[:8]}"

        keywords = set()
        for ex in pattern.examples[:5]:
            keywords.update(extract_keywords_from_question(ex.get("question", "")))

        rule_text = f"Use table '{pattern.correct_value}' instead of '{pattern.wrong_value}'. The table '{pattern.wrong_value}' does not exist."

        auto_fix = {
            "pattern": rf"\b{re.escape(pattern.wrong_value)}\b",
            "replacement": pattern.correct_value
        }

        example = None
        if pattern.examples:
            ex = pattern.examples[0]
            example = {
                "question": ex.get("question", ""),
                "sql": ex.get("corrected", "")
            }

        return GeneratedRule(
            id=rule_id,
            database=database,
            description=f"Table correction: {pattern.wrong_value} → {pattern.correct_value}",
            type="constraint",
            trigger_keywords=list(keywords)[:10],
            trigger_tables=[pattern.correct_value],
            rule_text=rule_text,
            auto_fix=auto_fix,
            example=example,
            priority="high",
            source="feedback_correction"
        )

    return None


@task(
    name="store_rule",
    description="Store generated rule in MongoDB",
    retries=2,
    tags=["feedback", "storage"]
)
async def store_rule_task(
    rule: GeneratedRule,
    correction_ids: List[str]
) -> bool:
    """Store a generated rule and mark corrections as processed."""
    logger = get_run_logger()

    import sys
    sys.path.insert(0, '/mnt/c/Projects/llm_website-usingSQLFeedBack/python_services')

    from mongodb import get_mongodb_service

    mongo = await get_mongodb_service()

    # Check if similar rule exists
    existing = await mongo.db["sql_rules"].find_one({
        "database": {"$regex": rule.database, "$options": "i"},
        "$or": [
            {"id": rule.id},
            {
                "auto_fix.pattern": rule.auto_fix.get("pattern") if rule.auto_fix else None,
                "auto_fix.replacement": rule.auto_fix.get("replacement") if rule.auto_fix else None
            }
        ]
    })

    if existing:
        logger.info(f"Similar rule already exists: {existing.get('id')}")
        # Still mark corrections as processed
        for cid in correction_ids:
            await mongo.db["agent_corrections"].update_one(
                {"_id": cid},
                {"$set": {"converted_to_rule": True, "rule_id": existing.get("id")}}
            )
        return False

    # Store new rule
    rule_doc = {
        "id": rule.id,
        "database": rule.database,
        "description": rule.description,
        "type": rule.type,
        "trigger_keywords": rule.trigger_keywords,
        "trigger_tables": rule.trigger_tables,
        "rule_text": rule.rule_text,
        "auto_fix": rule.auto_fix,
        "example": rule.example,
        "priority": rule.priority,
        "source": rule.source,
        "correction_ids": [str(cid) for cid in correction_ids],
        "active": True,
        "created_at": datetime.now(timezone.utc)
    }

    await mongo.db["sql_rules"].insert_one(rule_doc)
    logger.info(f"Created rule: {rule.id}")

    # Mark corrections as processed
    for cid in correction_ids:
        await mongo.db["agent_corrections"].update_one(
            {"_id": cid},
            {"$set": {"converted_to_rule": True, "rule_id": rule.id}}
        )

    return True


@task(
    name="create_exact_match_rule",
    description="Create exact match rule from high-confidence correction",
    tags=["feedback", "exact-match"]
)
async def create_exact_match_rule_task(
    correction: Dict,
    database: str
) -> Optional[str]:
    """Create an exact-match rule for high-confidence corrections."""
    logger = get_run_logger()

    import sys
    sys.path.insert(0, '/mnt/c/Projects/llm_website-usingSQLFeedBack/python_services')

    from mongodb import get_mongodb_service

    question = correction.get("natural_language", "")
    corrected_sql = correction.get("corrected_sql", "")
    correction_id = correction.get("_id")

    if not question or not corrected_sql:
        return None

    mongo = await get_mongodb_service()

    # Check for existing exact match
    question_hash = hashlib.md5(question.lower().strip().encode()).hexdigest()
    existing = await mongo.db["sql_rules"].find_one({
        "database": {"$regex": database, "$options": "i"},
        "example.question_hash": question_hash
    })

    if existing:
        logger.info(f"Exact match rule already exists for question")
        return existing.get("id")

    # Create exact match rule
    rule_id = f"exact_{question_hash[:8]}"

    rule_doc = {
        "id": rule_id,
        "database": database,
        "description": f"Exact match: {question[:50]}...",
        "type": "exact_match",
        "trigger_keywords": extract_keywords_from_question(question),
        "trigger_tables": extract_tables_from_sql(corrected_sql),
        "rule_text": "Use the exact SQL provided in the example.",
        "example": {
            "question": question,
            "question_hash": question_hash,
            "sql": corrected_sql
        },
        "priority": "highest",
        "source": "feedback_exact_match",
        "correction_id": str(correction_id),
        "active": True,
        "created_at": datetime.now(timezone.utc)
    }

    await mongo.db["sql_rules"].insert_one(rule_doc)

    # Mark correction as processed
    await mongo.db["agent_corrections"].update_one(
        {"_id": correction_id},
        {"$set": {"converted_to_rule": True, "rule_id": rule_id}}
    )

    logger.info(f"Created exact match rule: {rule_id}")
    return rule_id


@flow(
    name="feedback_to_rules",
    description="Convert user corrections to SQL rules",
    retries=1,
    log_prints=True
)
async def feedback_to_rules_flow(
    database: str = "EWRCentral",
    max_corrections: int = 100
) -> Dict[str, int]:
    """
    Main flow to convert feedback corrections into SQL rules.

    This flow:
    1. Fetches verified corrections that haven't been processed
    2. Analyzes each correction for patterns
    3. Groups similar patterns together
    4. Generates rules from patterns
    5. Creates exact-match rules for high-value corrections
    6. Stores rules and marks corrections as processed

    Args:
        database: Target database
        max_corrections: Maximum corrections to process

    Returns:
        Dictionary with counts of rules created
    """
    logger = get_run_logger()

    logger.info("=" * 60)
    logger.info("FEEDBACK TO RULES CONVERTER")
    logger.info(f"Database: {database}")
    logger.info("=" * 60)

    # Fetch unprocessed corrections
    corrections = await fetch_unprocessed_corrections_task(database, max_corrections)

    if not corrections:
        logger.info("No unprocessed corrections found")
        return {"patterns_found": 0, "rules_created": 0, "exact_matches": 0}

    # Analyze each correction
    patterns = []
    for correction in corrections:
        pattern = await analyze_correction_task(correction)
        if pattern:
            patterns.append(pattern)

    # Group similar patterns
    grouped_patterns = await group_patterns_task(patterns)

    # Generate and store rules from patterns
    rules_created = 0
    for pattern in grouped_patterns:
        rule = await generate_rule_from_pattern_task(pattern, database)
        if rule:
            # Get correction IDs from pattern examples
            correction_ids = [
                c.get("_id") for c in corrections
                if c.get("natural_language") in [ex.get("question") for ex in pattern.examples]
            ]

            success = await store_rule_task(rule, correction_ids)
            if success:
                rules_created += 1

    # Create exact match rules for corrections with high confidence
    exact_matches = 0
    for correction in corrections:
        # Create exact match for all verified corrections
        if correction.get("verified") and not correction.get("converted_to_rule"):
            rule_id = await create_exact_match_rule_task(correction, database)
            if rule_id:
                exact_matches += 1

    # Create summary artifact
    summary = f"""# Feedback to Rules Conversion Report

**Database:** {database}
**Processed:** {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}

## Results

| Metric | Count |
|--------|-------|
| Corrections Processed | {len(corrections)} |
| Patterns Identified | {len(grouped_patterns)} |
| Pattern Rules Created | {rules_created} |
| Exact Match Rules Created | {exact_matches} |
| **Total Rules** | **{rules_created + exact_matches}** |

## Pattern Types

"""

    pattern_counts = {}
    for p in grouped_patterns:
        pt = p.pattern_type
        pattern_counts[pt] = pattern_counts.get(pt, 0) + 1

    for pt, count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
        summary += f"- {pt}: {count}\n"

    await create_markdown_artifact(
        key="feedback-conversion-report",
        markdown=summary,
        description="Feedback to Rules Conversion Report"
    )

    logger.info("=" * 60)
    logger.info("CONVERSION COMPLETE")
    logger.info(f"Patterns found: {len(grouped_patterns)}")
    logger.info(f"Pattern rules created: {rules_created}")
    logger.info(f"Exact match rules: {exact_matches}")
    logger.info("=" * 60)

    return {
        "patterns_found": len(grouped_patterns),
        "rules_created": rules_created,
        "exact_matches": exact_matches
    }


if __name__ == "__main__":
    asyncio.run(feedback_to_rules_flow(
        database="EWRCentral",
        max_corrections=50
    ))
