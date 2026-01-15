"""
MongoDB Service - Ticket Matching Mixin

Handles ticket match history and phone customer mapping operations.
"""
import uuid
from datetime import datetime
from typing import Optional, List, Dict, TYPE_CHECKING

from pymongo import DESCENDING

from config import (
    COLLECTION_TICKET_MATCH_HISTORY,
    COLLECTION_PHONE_CUSTOMER_MAP,
    COLLECTION_AUDIO_ANALYSIS
)

if TYPE_CHECKING:
    from .base import MongoDBBase


class TicketMatchingMixin:
    """
    Mixin providing ticket matching and phone mapping operations.

    Methods:
        store_ticket_match_history: Store ticket match history for ML training
        get_match_history_for_analysis: Get match history for an analysis
        update_match_history_selection: Update match history with user selection
        link_ticket_to_analysis: Link a ticket to an audio analysis
        unlink_ticket_from_analysis: Remove ticket link from analysis
        store_phone_customer_mapping: Store phone to customer mapping
        lookup_customer_by_mapped_phone: Look up customer by phone
        get_phone_mappings_for_customer: Get all phone mappings for customer
    """

    async def store_ticket_match_history(
        self: 'MongoDBBase',
        analysis_id: str,
        match_method: str,
        search_text: str,
        search_text_type: str,
        candidates: List[Dict],
        selected_ticket_id: Optional[int] = None,
        auto_linked: bool = False,
        auto_link_confidence: Optional[float] = None,
        user_override: bool = False,
        feedback: Optional[str] = None
    ) -> Dict:
        """
        Store a ticket match history record for ML training.

        Args:
            analysis_id: Audio analysis ID
            match_method: Method used ('semantic', 'phone', 'combined')
            search_text: Text used for semantic search
            search_text_type: Type of text ('summary' or 'transcription')
            candidates: List of candidate tickets with scores
            selected_ticket_id: Final selected ticket ID
            auto_linked: Whether this was auto-linked
            auto_link_confidence: Confidence score if auto-linked
            user_override: Whether user changed auto-link selection
            feedback: User feedback on match quality
        """
        if not self.is_initialized:
            await self.initialize()

        collection = self.db[COLLECTION_TICKET_MATCH_HISTORY]

        record_id = str(uuid.uuid4())
        now = datetime.utcnow()

        record = {
            "id": record_id,
            "analysis_id": analysis_id,
            "matched_at": now,
            "match_method": match_method,
            "search_text": search_text[:2000],  # Truncate for storage
            "search_text_type": search_text_type,
            "candidates": candidates,
            "selected_ticket_id": selected_ticket_id,
            "auto_linked": auto_linked,
            "auto_link_confidence": auto_link_confidence,
            "user_override": user_override,
            "feedback": feedback
        }

        await collection.insert_one(record)

        return {
            "success": True,
            "history_id": record_id,
            "message": "Match history stored successfully"
        }

    async def get_match_history_for_analysis(
        self: 'MongoDBBase',
        analysis_id: str
    ) -> List[Dict]:
        """Get all match history records for an analysis"""
        if not self.is_initialized:
            await self.initialize()

        collection = self.db[COLLECTION_TICKET_MATCH_HISTORY]

        cursor = collection.find({"analysis_id": analysis_id}).sort("matched_at", -1)
        records = await cursor.to_list(length=100)

        for record in records:
            record.pop("_id", None)
            if record.get("matched_at"):
                record["matched_at"] = record["matched_at"].isoformat()

        return records

    async def update_match_history_selection(
        self: 'MongoDBBase',
        history_id: str,
        selected_ticket_id: Optional[int],
        user_override: bool = True,
        feedback: Optional[str] = None
    ) -> bool:
        """Update a match history record when user makes a selection"""
        if not self.is_initialized:
            await self.initialize()

        collection = self.db[COLLECTION_TICKET_MATCH_HISTORY]

        update_fields = {
            "selected_ticket_id": selected_ticket_id,
            "user_override": user_override
        }
        if feedback is not None:
            update_fields["feedback"] = feedback

        result = await collection.update_one(
            {"id": history_id},
            {"$set": update_fields}
        )

        return result.modified_count > 0 or result.matched_count > 0

    async def link_ticket_to_analysis(
        self: 'MongoDBBase',
        analysis_id: str,
        ticket_id: int
    ) -> bool:
        """
        Link a ticket to an audio analysis.

        Updates the linked_ticket_id field on the audio analysis.
        Also creates a phone-to-customer mapping if phone number is available.
        """
        if not self.is_initialized:
            await self.initialize()

        collection = self.db[COLLECTION_AUDIO_ANALYSIS]

        result = await collection.update_one(
            {"id": analysis_id},
            {"$set": {"linked_ticket_id": ticket_id, "updated_at": datetime.utcnow()}}
        )

        # Try to create phone-to-customer mapping
        try:
            analysis = await self.get_audio_analysis(analysis_id)

            if analysis and analysis.get("call_metadata", {}).get("phone_number"):
                phone_number = analysis["call_metadata"]["phone_number"]

                # Import here to avoid circular imports
                from services.ewrcentral_agent import EWRCentralAgent
                agent = EWRCentralAgent()

                query = f"""
                SELECT ct.CentralCompanyID
                FROM CentralTickets ct
                WHERE ct.CentralTicketID = {ticket_id}
                """

                ticket_result = await agent.execute_query(query)

                if ticket_result.get("success") and ticket_result.get("data"):
                    company_id = ticket_result["data"][0].get("CentralCompanyID")

                    if company_id:
                        await self.store_phone_customer_mapping(
                            phone_number=phone_number,
                            customer_id=company_id,
                            source="ticket_link"
                        )
        except Exception as e:
            print(f"Warning: Failed to create phone mapping for analysis {analysis_id}: {e}")

        return result.modified_count > 0 or result.matched_count > 0

    async def unlink_ticket_from_analysis(self: 'MongoDBBase', analysis_id: str) -> bool:
        """Remove ticket link from an audio analysis"""
        if not self.is_initialized:
            await self.initialize()

        collection = self.db[COLLECTION_AUDIO_ANALYSIS]

        result = await collection.update_one(
            {"id": analysis_id},
            {"$set": {"linked_ticket_id": None, "updated_at": datetime.utcnow()}}
        )

        return result.modified_count > 0 or result.matched_count > 0

    # ============================================================================
    # Phone Customer Mapping Methods
    # ============================================================================

    def _normalize_phone_number(self, phone_number: str) -> str:
        """
        Normalize phone number for consistent storage and lookup.

        Examples:
            "(843) 858-0749" -> "8438580749"
            "+1-843-858-0749" -> "18438580749"
        """
        normalized = ''.join(filter(str.isdigit, phone_number))
        return normalized

    async def store_phone_customer_mapping(
        self: 'MongoDBBase',
        phone_number: str,
        customer_id: int,
        source: str = "ticket_link"
    ) -> Dict:
        """
        Store or update a phone number to customer ID mapping.

        Args:
            phone_number: Phone number (will be normalized)
            customer_id: EWR Customer ID
            source: How the mapping was discovered

        Returns:
            Dict with operation result
        """
        if not self.is_initialized:
            await self.initialize()

        collection = self.db[COLLECTION_PHONE_CUSTOMER_MAP]

        normalized_phone = self._normalize_phone_number(phone_number)

        if not normalized_phone:
            return {
                "success": False,
                "message": "Invalid phone number - no digits found"
            }

        now = datetime.utcnow()

        result = await collection.update_one(
            {
                "phone_number": normalized_phone,
                "customer_id": customer_id
            },
            {
                "$set": {
                    "last_seen": now,
                    "source": source
                },
                "$setOnInsert": {
                    "first_seen": now
                },
                "$inc": {
                    "occurrence_count": 1
                }
            },
            upsert=True
        )

        return {
            "success": True,
            "phone_number": normalized_phone,
            "customer_id": customer_id,
            "new_mapping": result.upserted_id is not None,
            "message": "Phone mapping created" if result.upserted_id else "Phone mapping updated"
        }

    async def lookup_customer_by_mapped_phone(
        self: 'MongoDBBase',
        phone_number: str
    ) -> Optional[Dict]:
        """
        Look up customer ID by phone number.

        Args:
            phone_number: Phone number to look up (will be normalized)

        Returns:
            Mapping document if found, None otherwise
        """
        if not self.is_initialized:
            await self.initialize()

        collection = self.db[COLLECTION_PHONE_CUSTOMER_MAP]

        normalized_phone = self._normalize_phone_number(phone_number)

        if not normalized_phone:
            return None

        mapping = await collection.find_one({"phone_number": normalized_phone})

        if not mapping:
            return None

        mapping.pop("_id", None)

        return mapping

    async def get_phone_mappings_for_customer(
        self: 'MongoDBBase',
        customer_id: int
    ) -> List[Dict]:
        """
        Get all phone number mappings for a customer.

        Args:
            customer_id: EWR Customer ID

        Returns:
            List of phone mapping documents
        """
        if not self.is_initialized:
            await self.initialize()

        collection = self.db[COLLECTION_PHONE_CUSTOMER_MAP]

        cursor = collection.find(
            {"customer_id": customer_id}
        ).sort("last_seen", DESCENDING)

        mappings = []
        async for mapping in cursor:
            mapping.pop("_id", None)
            mappings.append(mapping)

        return mappings
