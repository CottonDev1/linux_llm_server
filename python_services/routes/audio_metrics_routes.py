"""Audio metrics routes for staff performance metrics."""
from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import Optional, Dict

from mongodb import get_mongodb_service
from log_service import log_pipeline, log_error
from config import COLLECTION_AUDIO_ANALYSIS

router = APIRouter(prefix="/audio", tags=["Audio Analysis"])


class StaffMetricsResponse(BaseModel):
    """Response model for staff metrics"""
    success: bool
    staff_name: str
    # MongoDB metrics
    calls_analyzed: int = 0
    average_mood: Optional[str] = None
    mood_counts: Dict[str, int] = {}
    average_call_length_seconds: float = 0
    average_call_length_formatted: str = "0:00"
    total_call_time_seconds: float = 0
    total_call_time_formatted: str = "0:00"
    # SQL metrics
    tickets_entered_30d: int = 0
    tickets_closed_30d: int = 0
    user_id: Optional[int] = None
    error: Optional[str] = None


@router.get("/staff-metrics/{staff_name}", response_model=StaffMetricsResponse)
async def get_staff_metrics(staff_name: str, request: Request, days: int = 30):
    """
    Get comprehensive metrics for a staff member.

    Combines:
    - MongoDB call analysis data (calls analyzed, mood, call length)
    - EWRCentral ticket data (tickets entered/closed in specified days window)

    Args:
        staff_name: Name of the staff member
        days: Number of days to look back for ticket data (default: 30)
    """
    user_ip = request.client.host if request.client else "Unknown"

    log_pipeline("AUDIO", user_ip, "Getting staff metrics", {"staff_name": staff_name})

    result = StaffMetricsResponse(success=True, staff_name=staff_name)

    try:
        # =====================================================================
        # MongoDB Metrics - Call Analysis Data
        # =====================================================================
        mongodb = get_mongodb_service()
        if mongodb and mongodb.is_initialized:
            collection = mongodb.db[COLLECTION_AUDIO_ANALYSIS]

            # Find all calls for this staff member
            # Match on customer_support_staff field (case-insensitive)
            # Support matching on full name OR first name only (e.g., "Christine Harpster" matches "Christine")
            first_name = staff_name.split()[0] if staff_name else staff_name
            pipeline = [
                {
                    "$match": {
                        "$or": [
                            {"customer_support_staff": {"$regex": f"^{staff_name}$", "$options": "i"}},
                            {"metadata.customer_support_staff": {"$regex": f"^{staff_name}$", "$options": "i"}},
                            {"customer_support_staff": {"$regex": f"^{first_name}$", "$options": "i"}},
                            {"metadata.customer_support_staff": {"$regex": f"^{first_name}$", "$options": "i"}}
                        ]
                    }
                },
                {
                    "$group": {
                        "_id": None,
                        "count": {"$sum": 1},
                        "moods": {"$push": "$mood"},
                        "total_duration": {"$sum": "$audio_metadata.duration_seconds"}
                    }
                }
            ]

            agg_result = await collection.aggregate(pipeline).to_list(length=1)

            if agg_result:
                data = agg_result[0]
                result.calls_analyzed = data.get("count", 0)

                # Calculate average call length
                total_seconds = data.get("total_duration", 0)
                if result.calls_analyzed > 0:
                    avg_seconds = total_seconds / result.calls_analyzed
                    result.average_call_length_seconds = avg_seconds
                    mins = int(avg_seconds // 60)
                    secs = int(avg_seconds % 60)
                    result.average_call_length_formatted = f"{mins}:{secs:02d}"

                # Calculate total call time
                result.total_call_time_seconds = total_seconds
                total_mins = int(total_seconds // 60)
                total_secs = int(total_seconds % 60)
                if total_mins >= 60:
                    hours = int(total_mins // 60)
                    remaining_mins = int(total_mins % 60)
                    result.total_call_time_formatted = f"{hours}h {remaining_mins}m"
                else:
                    result.total_call_time_formatted = f"{total_mins}:{total_secs:02d}"

                # Calculate mood distribution and average
                moods = [m for m in data.get("moods", []) if m]
                if moods:
                    mood_counts = {}
                    for mood in moods:
                        mood_upper = mood.upper() if mood else "UNKNOWN"
                        mood_counts[mood_upper] = mood_counts.get(mood_upper, 0) + 1
                    result.mood_counts = mood_counts

                    # Find most common mood
                    if mood_counts:
                        result.average_mood = max(mood_counts, key=mood_counts.get)

        # =====================================================================
        # SQL Metrics - EWRCentral Ticket Data
        # =====================================================================
        import pymssql

        connection = pymssql.connect(
            server='EWRSQLPROD',
            database='EWRCentral',
            user='EWR\\chad.walker',
            password='6454@@Christina',
            port='1433'
        )
        cursor = connection.cursor(as_dict=True)

        # First, find the user ID by name
        # Split staff_name and search for matching user
        name_parts = staff_name.strip().split()
        if len(name_parts) >= 2:
            first_name = name_parts[0]
            last_name = name_parts[-1]
            cursor.execute("""
                SELECT TOP 1 CentralUserID
                FROM CentralUsers
                WHERE FirstName LIKE %s AND LastName LIKE %s AND IsActive = 1
                ORDER BY LastUpdateUTC DESC
            """, (f"{first_name}%", f"{last_name}%"))
        else:
            # Single name - search first or last
            cursor.execute("""
                SELECT TOP 1 CentralUserID
                FROM CentralUsers
                WHERE (FirstName LIKE %s OR LastName LIKE %s) AND IsActive = 1
                ORDER BY LastUpdateUTC DESC
            """, (f"{staff_name}%", f"{staff_name}%"))

        user_row = cursor.fetchone()

        if user_row:
            result.user_id = user_row['CentralUserID']

            # Count tickets entered in the specified time window (where user created the ticket)
            cursor.execute(f"""
                SELECT COUNT(*) AS ticket_count
                FROM CentralTickets
                WHERE AddCentralUserID = %s
                  AND AddTicketDate >= DATEADD(day, -{days}, GETDATE())
            """, (result.user_id,))
            entered_row = cursor.fetchone()
            result.tickets_entered_30d = entered_row['ticket_count'] if entered_row else 0

            # Count tickets closed in the specified time window
            # A ticket is considered "closed" by a user if they were the last one to update it
            # and the ticket is now in a closed status
            cursor.execute(f"""
                SELECT COUNT(DISTINCT cta.CentralTicketID) AS closed_count
                FROM CentralTicketAudits cta
                INNER JOIN (
                    SELECT CentralTicketID, MAX(CentralTicketAuditID) AS LastAuditID
                    FROM CentralTicketAudits
                    WHERE AuditDateTime >= DATEADD(day, -{days}, GETDATE())
                    GROUP BY CentralTicketID
                ) last_audit ON cta.CentralTicketAuditID = last_audit.LastAuditID
                INNER JOIN CentralTickets ct ON cta.CentralTicketID = ct.CentralTicketID
                WHERE cta.ModifiedByCentralUserID = %s
                  AND ct.TicketStatusTypeID IN (
                      SELECT TypeID FROM Types WHERE Description LIKE '%Closed%' OR Description LIKE '%Complete%' OR Description LIKE '%Resolved%'
                  )
            """, (result.user_id,))
            closed_row = cursor.fetchone()
            result.tickets_closed_30d = closed_row['closed_count'] if closed_row else 0

        cursor.close()
        connection.close()

        log_pipeline("AUDIO", user_ip, "Staff metrics retrieved", {
            "staff_name": staff_name,
            "calls_analyzed": result.calls_analyzed,
            "tickets_entered": result.tickets_entered_30d,
            "tickets_closed": result.tickets_closed_30d
        })

        return result

    except Exception as e:
        log_error("AUDIO", user_ip, f"Failed to get staff metrics for {staff_name}", str(e))
        return StaffMetricsResponse(
            success=False,
            staff_name=staff_name,
            error=str(e)
        )
