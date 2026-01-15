"""
Prefect Ticket Matching Pipeline

Orchestrates semantic ticket matching for audio analyses:
1. Retrieve Analysis - Load audio analysis from MongoDB
2. Query Candidates - Get tickets from call date range
3. Compute Embeddings - Embed summary/transcription and ticket notes
4. Score Candidates - Combined semantic + metadata scoring
5. Store History - Record match attempt for training

Features:
- Multi-agent coordination through Prefect tasks
- Match history for ML training
- Configurable scoring weights
- Auto-linking above confidence threshold
- Built-in retries for resilience
"""

import asyncio
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
import numpy as np

from prefect import flow, task, get_run_logger
from prefect.artifacts import create_markdown_artifact


@dataclass
class AnalysisData:
    """Audio analysis data for matching"""
    analysis_id: str
    summary: str = ""
    transcription: str = ""
    call_date: Optional[datetime] = None
    phone_number: Optional[str] = None
    extension: Optional[str] = None
    staff_id: Optional[str] = None
    direction: str = "Unknown"
    success: bool = True
    error: str = ""


@dataclass
class TicketCandidate:
    """Candidate ticket from SQL query"""
    ticket_id: int
    ticket_title: str = ""
    main_note: str = ""
    all_notes: str = ""
    add_date: Optional[datetime] = None
    phone_number: Optional[str] = None
    user_id: Optional[int] = None


@dataclass
class ScoredCandidate:
    """Ticket with computed scores"""
    ticket_id: int
    ticket_title: str = ""
    semantic_score: float = 0.0
    combined_score: float = 0.0
    match_reasons: List[str] = field(default_factory=list)
    phone_match: bool = False
    staff_match: bool = False
    time_proximity_minutes: Optional[int] = None


@dataclass
class MatchResult:
    """Final matching result"""
    success: bool = True
    analysis_id: str = ""
    search_text_type: str = ""
    total_candidates: int = 0
    matches: List[ScoredCandidate] = field(default_factory=list)
    best_match: Optional[ScoredCandidate] = None
    auto_linked: bool = False
    linked_ticket_id: Optional[int] = None
    duration_seconds: float = 0.0
    error: str = ""


# Scoring weights
SCORING_WEIGHTS = {
    'semantic': 0.40,
    'phone': 0.25,
    'time': 0.20,
    'staff': 0.15
}


@task(
    name="retrieve_audio_analysis",
    description="Load audio analysis data from MongoDB",
    retries=2,
    retry_delay_seconds=10,
    tags=["ticket-matching", "mongodb"]
)
async def retrieve_analysis_task(analysis_id: str) -> AnalysisData:
    """
    Retrieve audio analysis from MongoDB.

    Args:
        analysis_id: MongoDB document ID

    Returns:
        AnalysisData with summary, transcription, and metadata
    """
    logger = get_run_logger()
    logger.info(f"Retrieving analysis: {analysis_id}")

    result = AnalysisData(analysis_id=analysis_id)

    try:
        import sys
        sys.path.insert(0, '..')
        from mongodb import MongoDBService

        mongo = MongoDBService()
        await mongo.connect()

        analysis = await mongo.get_audio_analysis_by_id(analysis_id)

        if not analysis:
            result.success = False
            result.error = f"Analysis not found: {analysis_id}"
            logger.error(result.error)
            return result

        result.summary = analysis.get("transcription_summary", "")
        result.transcription = analysis.get("transcription", "")

        # Parse call metadata
        call_meta = analysis.get("call_metadata", {})
        result.direction = call_meta.get("direction", "Unknown")
        result.phone_number = call_meta.get("phone_number")
        result.extension = call_meta.get("extension")

        # Get call date
        if call_meta.get("call_date"):
            result.call_date = datetime.fromisoformat(call_meta["call_date"].replace("Z", "+00:00"))
        elif analysis.get("analyzed_at"):
            analyzed_at = analysis["analyzed_at"]
            if isinstance(analyzed_at, str):
                result.call_date = datetime.fromisoformat(analyzed_at.replace("Z", "+00:00"))
            else:
                result.call_date = analyzed_at

        logger.info(f"Analysis retrieved: {len(result.summary)} char summary, {len(result.transcription)} char transcription")
        logger.info(f"Call date: {result.call_date}, Phone: {result.phone_number}")

        await mongo.close()

    except Exception as e:
        logger.error(f"Failed to retrieve analysis: {e}")
        result.success = False
        result.error = str(e)

    return result


@task(
    name="query_candidate_tickets",
    description="Query tickets from EWRCentral for date range",
    retries=2,
    retry_delay_seconds=15,
    tags=["ticket-matching", "sql", "ewrcentral"]
)
async def query_candidates_task(
    analysis: AnalysisData,
    days_window: int = 3
) -> List[TicketCandidate]:
    """
    Query candidate tickets from EWRCentral.

    Args:
        analysis: Audio analysis with call date
        days_window: Days before/after to search (default 3)

    Returns:
        List of TicketCandidate objects
    """
    logger = get_run_logger()

    if not analysis.success or not analysis.call_date:
        logger.warning("No valid analysis or call date for ticket query")
        return []

    start_date = analysis.call_date - timedelta(days=days_window)
    end_date = analysis.call_date + timedelta(days=days_window)

    logger.info(f"Querying tickets from {start_date.date()} to {end_date.date()}")

    candidates = []

    try:
        import sys
        sys.path.insert(0, '..')
        from services.ewrcentral_agent import EWRCentralAgent

        agent = EWRCentralAgent()

        # Query tickets in date range
        query = f"""
        SELECT TOP 50
            ct.CentralTicketID,
            ct.TicketTitle,
            ct.Note AS MainNote,
            ct.AddTicketDate,
            ct.CustomerContactPhoneNumber,
            ct.AddCentralUserID,
            (
                SELECT STRING_AGG(ctn.Note, ' || ') WITHIN GROUP (ORDER BY ctn.AddNoteDate)
                FROM CentralTicketNotes ctn
                WHERE ctn.CentralTicketID = ct.CentralTicketID
            ) AS AllNotes
        FROM CentralTickets ct
        WHERE ct.AddTicketDate BETWEEN '{start_date.strftime('%Y-%m-%d')}' AND '{end_date.strftime('%Y-%m-%d')}'
        ORDER BY ct.AddTicketDate DESC
        """

        result = await agent.execute_query(query)

        if result.get("success") and result.get("data"):
            for row in result["data"]:
                candidate = TicketCandidate(
                    ticket_id=row.get("CentralTicketID"),
                    ticket_title=row.get("TicketTitle", ""),
                    main_note=row.get("MainNote", "") or "",
                    all_notes=row.get("AllNotes", "") or "",
                    add_date=row.get("AddTicketDate"),
                    phone_number=row.get("CustomerContactPhoneNumber"),
                    user_id=row.get("AddCentralUserID")
                )
                candidates.append(candidate)

        logger.info(f"Found {len(candidates)} candidate tickets")

    except Exception as e:
        logger.error(f"Failed to query tickets: {e}")

    return candidates


@task(
    name="compute_embeddings",
    description="Generate embeddings for semantic comparison",
    retries=1,
    retry_delay_seconds=10,
    tags=["ticket-matching", "embeddings", "ml"]
)
async def compute_embeddings_task(
    search_text: str,
    candidates: List[TicketCandidate]
) -> Dict[str, Any]:
    """
    Compute embeddings for search text and ticket notes.

    Args:
        search_text: Summary or transcription to match
        candidates: Tickets to compare against

    Returns:
        Dict with search_embedding and ticket_embeddings
    """
    logger = get_run_logger()
    logger.info(f"Computing embeddings for {len(search_text)} char text and {len(candidates)} tickets")

    result = {
        "search_embedding": None,
        "ticket_embeddings": {},
        "success": True
    }

    if not search_text or not candidates:
        return result

    try:
        from sentence_transformers import SentenceTransformer

        # Load model (cached after first load)
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # Embed search text
        result["search_embedding"] = model.encode(search_text).tolist()

        # Embed ticket notes
        for candidate in candidates:
            combined_text = f"{candidate.ticket_title} {candidate.main_note} {candidate.all_notes}".strip()
            if combined_text:
                embedding = model.encode(combined_text).tolist()
                result["ticket_embeddings"][candidate.ticket_id] = embedding

        logger.info(f"Computed {len(result['ticket_embeddings'])} ticket embeddings")

    except Exception as e:
        logger.error(f"Embedding computation failed: {e}")
        result["success"] = False

    return result


@task(
    name="score_candidates",
    description="Score candidates with combined signals",
    retries=1,
    tags=["ticket-matching", "scoring"]
)
async def score_candidates_task(
    analysis: AnalysisData,
    candidates: List[TicketCandidate],
    embeddings: Dict[str, Any],
    threshold: float = 0.5
) -> List[ScoredCandidate]:
    """
    Score candidates using semantic + metadata signals.

    Args:
        analysis: Audio analysis data
        candidates: Ticket candidates
        embeddings: Computed embeddings
        threshold: Minimum combined score

    Returns:
        List of ScoredCandidate sorted by score
    """
    logger = get_run_logger()
    logger.info(f"Scoring {len(candidates)} candidates")

    scored = []
    search_embedding = embeddings.get("search_embedding")
    ticket_embeddings = embeddings.get("ticket_embeddings", {})

    if not search_embedding:
        logger.warning("No search embedding available")
        return scored

    search_vec = np.array(search_embedding)

    for candidate in candidates:
        ticket_embedding = ticket_embeddings.get(candidate.ticket_id)

        # Compute semantic score
        semantic_score = 0.0
        if ticket_embedding:
            ticket_vec = np.array(ticket_embedding)
            # Cosine similarity
            dot = np.dot(search_vec, ticket_vec)
            norm = np.linalg.norm(search_vec) * np.linalg.norm(ticket_vec)
            if norm > 0:
                semantic_score = float(dot / norm)

        # Check phone match
        phone_match = False
        if analysis.phone_number and candidate.phone_number:
            # Normalize phone numbers for comparison
            a_phone = ''.join(c for c in analysis.phone_number if c.isdigit())
            c_phone = ''.join(c for c in candidate.phone_number if c.isdigit())
            phone_match = a_phone[-10:] == c_phone[-10:] if len(a_phone) >= 10 and len(c_phone) >= 10 else False

        # Compute time proximity
        time_proximity_minutes = None
        time_score = 0.0
        if analysis.call_date and candidate.add_date:
            delta = abs((analysis.call_date - candidate.add_date).total_seconds() / 60)
            time_proximity_minutes = int(delta)
            # Decay: full score at 0 min, half at 2 hours, zero at 24 hours
            time_score = max(0, 1 - (delta / (24 * 60)))

        # Staff match (placeholder - would need to map extension to user)
        staff_match = False

        # Combined score
        combined_score = (
            semantic_score * SCORING_WEIGHTS['semantic'] +
            (1.0 if phone_match else 0.0) * SCORING_WEIGHTS['phone'] +
            time_score * SCORING_WEIGHTS['time'] +
            (1.0 if staff_match else 0.0) * SCORING_WEIGHTS['staff']
        )

        # Build match reasons
        match_reasons = []
        if semantic_score > 0.3:
            match_reasons.append("semantic")
        if phone_match:
            match_reasons.append("phone")
        if time_proximity_minutes and time_proximity_minutes < 120:
            match_reasons.append("time")
        if staff_match:
            match_reasons.append("staff")

        if combined_score >= threshold:
            scored.append(ScoredCandidate(
                ticket_id=candidate.ticket_id,
                ticket_title=candidate.ticket_title,
                semantic_score=semantic_score,
                combined_score=combined_score,
                match_reasons=match_reasons,
                phone_match=phone_match,
                staff_match=staff_match,
                time_proximity_minutes=time_proximity_minutes
            ))

    # Sort by combined score descending
    scored.sort(key=lambda x: x.combined_score, reverse=True)

    logger.info(f"Found {len(scored)} matches above threshold {threshold}")

    return scored


@task(
    name="store_match_history",
    description="Store match attempt for training",
    retries=2,
    retry_delay_seconds=10,
    tags=["ticket-matching", "history", "mongodb"]
)
async def store_history_task(
    analysis_id: str,
    search_text: str,
    search_text_type: str,
    scored_candidates: List[ScoredCandidate],
    auto_linked: bool,
    linked_ticket_id: Optional[int]
) -> bool:
    """
    Store match attempt in history collection.

    Args:
        analysis_id: Audio analysis ID
        search_text: Text used for matching
        search_text_type: "summary" or "transcription"
        scored_candidates: All scored candidates
        auto_linked: Whether auto-link was triggered
        linked_ticket_id: Linked ticket if any

    Returns:
        True if stored successfully
    """
    logger = get_run_logger()
    logger.info(f"Storing match history for {analysis_id}")

    try:
        import sys
        sys.path.insert(0, '..')
        from mongodb import MongoDBService

        mongo = MongoDBService()
        await mongo.connect()

        # Convert scored candidates to dicts
        candidates_data = []
        for sc in scored_candidates[:20]:  # Store top 20
            candidates_data.append({
                "ticket_id": sc.ticket_id,
                "ticket_title": sc.ticket_title,
                "semantic_score": sc.semantic_score,
                "combined_score": sc.combined_score,
                "match_reasons": sc.match_reasons,
                "phone_match": sc.phone_match,
                "staff_match": sc.staff_match,
                "time_proximity_minutes": sc.time_proximity_minutes
            })

        await mongo.store_ticket_match_history(
            analysis_id=analysis_id,
            match_method="semantic",
            search_text=search_text[:1000],  # Truncate for storage
            search_text_type=search_text_type,
            candidates=candidates_data,
            auto_linked=auto_linked,
            auto_link_confidence=scored_candidates[0].combined_score if scored_candidates else None
        )

        await mongo.close()
        logger.info("Match history stored successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to store history: {e}")
        return False


@flow(
    name="ticket-matching-pipeline",
    description="Semantic ticket matching for audio analyses",
    retries=1,
    retry_delay_seconds=60
)
async def ticket_matching_flow(
    analysis_id: str,
    use_summary: bool = True,
    threshold: float = 0.5,
    auto_link_threshold: float = 0.75,
    auto_link: bool = False
) -> Dict[str, Any]:
    """
    Ticket Matching Pipeline - Find best ticket match for audio analysis.

    This flow:
    1. Retrieves audio analysis from MongoDB
    2. Queries candidate tickets from call date range
    3. Computes semantic embeddings
    4. Scores candidates with combined signals
    5. Stores match history for training

    Args:
        analysis_id: MongoDB document ID
        use_summary: Use summary (True) or transcription (False)
        threshold: Minimum score to include in results
        auto_link_threshold: Minimum score for auto-linking
        auto_link: Whether to auto-link above threshold

    Returns:
        Dict with matches, best_match, and metadata
    """
    logger = get_run_logger()
    flow_start = time.time()

    logger.info(f"Starting ticket matching for: {analysis_id}")
    logger.info(f"Settings: use_summary={use_summary}, threshold={threshold}, auto_link={auto_link}")

    result = MatchResult(analysis_id=analysis_id)

    # Step 1: Retrieve analysis
    analysis = await retrieve_analysis_task(analysis_id=analysis_id)

    if not analysis.success:
        result.success = False
        result.error = analysis.error
        return result.__dict__

    # Step 2: Query candidate tickets
    candidates = await query_candidates_task(analysis=analysis)
    result.total_candidates = len(candidates)

    if not candidates:
        logger.warning("No candidate tickets found")
        result.duration_seconds = time.time() - flow_start
        return result.__dict__

    # Step 3: Compute embeddings
    search_text = analysis.summary if use_summary and analysis.summary else analysis.transcription
    result.search_text_type = "summary" if use_summary and analysis.summary else "transcription"

    embeddings = await compute_embeddings_task(
        search_text=search_text,
        candidates=candidates
    )

    # Step 4: Score candidates
    scored = await score_candidates_task(
        analysis=analysis,
        candidates=candidates,
        embeddings=embeddings,
        threshold=threshold
    )

    result.matches = scored

    if scored:
        result.best_match = scored[0]

        # Auto-link if enabled and above threshold
        if auto_link and scored[0].combined_score >= auto_link_threshold:
            result.auto_linked = True
            result.linked_ticket_id = scored[0].ticket_id

            # Update MongoDB
            try:
                import sys
                sys.path.insert(0, '..')
                from mongodb import MongoDBService

                mongo = MongoDBService()
                await mongo.connect()
                await mongo.link_ticket_to_analysis(analysis_id, scored[0].ticket_id)
                await mongo.close()
                logger.info(f"Auto-linked ticket {scored[0].ticket_id}")
            except Exception as e:
                logger.error(f"Failed to auto-link: {e}")

    # Step 5: Store match history
    await store_history_task(
        analysis_id=analysis_id,
        search_text=search_text,
        search_text_type=result.search_text_type,
        scored_candidates=scored,
        auto_linked=result.auto_linked,
        linked_ticket_id=result.linked_ticket_id
    )

    result.duration_seconds = time.time() - flow_start

    # Create Prefect artifact
    matches_table = "\n".join([
        f"| {m.ticket_id} | {m.ticket_title[:30]}... | {m.semantic_score:.2f} | {m.combined_score:.2f} | {', '.join(m.match_reasons)} |"
        for m in scored[:10]
    ]) if scored else "No matches found"

    await create_markdown_artifact(
        key="ticket-match-result",
        markdown=f"""
# Ticket Matching Results

## Overview
- **Analysis ID**: {analysis_id}
- **Search Type**: {result.search_text_type}
- **Candidates Found**: {result.total_candidates}
- **Matches Above Threshold**: {len(scored)}
- **Processing Time**: {result.duration_seconds:.2f}s

## Best Match
{"**Ticket #" + str(result.best_match.ticket_id) + "**: " + result.best_match.ticket_title if result.best_match else "No match found"}
{"- Score: " + f"{result.best_match.combined_score:.2f}" if result.best_match else ""}
{"- Auto-linked: Yes" if result.auto_linked else ""}

## Top Matches

| Ticket ID | Title | Semantic | Combined | Reasons |
|-----------|-------|----------|----------|---------|
{matches_table}
        """,
        description=f"Ticket matching for {analysis_id}"
    )

    logger.info(f"Ticket matching complete in {result.duration_seconds:.2f}s")
    logger.info(f"Best match: {result.best_match.ticket_id if result.best_match else 'None'}")

    # Convert dataclasses to dicts for return
    return {
        "success": result.success,
        "analysis_id": result.analysis_id,
        "search_text_type": result.search_text_type,
        "total_candidates": result.total_candidates,
        "matches": [
            {
                "ticket_id": m.ticket_id,
                "ticket_title": m.ticket_title,
                "semantic_score": m.semantic_score,
                "combined_score": m.combined_score,
                "match_reasons": m.match_reasons,
                "phone_match": m.phone_match,
                "staff_match": m.staff_match,
                "time_proximity_minutes": m.time_proximity_minutes
            }
            for m in result.matches
        ],
        "best_match": {
            "ticket_id": result.best_match.ticket_id,
            "ticket_title": result.best_match.ticket_title,
            "semantic_score": result.best_match.semantic_score,
            "combined_score": result.best_match.combined_score,
            "match_reasons": result.best_match.match_reasons
        } if result.best_match else None,
        "auto_linked": result.auto_linked,
        "linked_ticket_id": result.linked_ticket_id,
        "duration_seconds": result.duration_seconds,
        "error": result.error
    }


def run_ticket_matching_flow(
    analysis_id: str,
    use_summary: bool = True,
    threshold: float = 0.5,
    auto_link_threshold: float = 0.75,
    auto_link: bool = False
) -> Dict[str, Any]:
    """
    Run the ticket matching flow synchronously.

    This is the main entry point for the API.

    Args:
        analysis_id: MongoDB document ID
        use_summary: Use summary or transcription
        threshold: Minimum score threshold
        auto_link_threshold: Threshold for auto-linking
        auto_link: Whether to auto-link

    Returns:
        Dict with matching results
    """
    return asyncio.run(ticket_matching_flow(
        analysis_id=analysis_id,
        use_summary=use_summary,
        threshold=threshold,
        auto_link_threshold=auto_link_threshold,
        auto_link=auto_link
    ))


# Export for use in __init__.py
__all__ = [
    "ticket_matching_flow",
    "run_ticket_matching_flow",
    "retrieve_analysis_task",
    "query_candidates_task",
    "compute_embeddings_task",
    "score_candidates_task",
    "store_history_task"
]
