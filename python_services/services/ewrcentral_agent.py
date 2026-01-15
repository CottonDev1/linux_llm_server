"""
EWRCentral SQL Agent Service

Provides a centralized service for executing queries against the EWRCentral database.
Supports both predefined queries (for known operations like staff lookup) and
dynamic SQL generation via the LLM SQL generator.

This service:
- Manages database connections to EWRSQLPROD
- Provides typed methods for common operations
- Can use the SQL generator for dynamic queries
- Handles connection pooling and error handling
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# EWRCentral database configuration
EWRCENTRAL_CONFIG = {
    "server": "EWRSQLPROD",
    "database": "EWRCentral",
    "user": "EWR\\chad.walker",
    "password": "6454@@Christina",
    "port": "1433"
}


@dataclass
class StaffInfo:
    """Staff member information from CentralUsers"""
    user_id: int
    first_name: str
    last_name: str
    full_name: str
    email: Optional[str] = None
    extension: Optional[str] = None
    found: bool = True


@dataclass
class CustomerInfo:
    """Customer information from EWRCentral database"""
    customer_name: Optional[str] = None
    company_name: Optional[str] = None
    company_id: Optional[int] = None
    phone_number: Optional[str] = None
    email: Optional[str] = None
    last_ticket_date: Optional[datetime] = None
    ticket_count: int = 0
    found: bool = False
    match_source: str = ""  # "ticket", "company", etc.


@dataclass
class TicketMatch:
    """A ticket that potentially matches a phone call"""
    ticket_id: int
    title: str
    note: Optional[str]
    created_date: datetime
    customer_name: Optional[str]
    customer_phone: Optional[str]
    company_name: Optional[str]
    created_by: str
    status: Optional[str]
    match_score: int = 0
    match_reasons: List[str] = field(default_factory=list)


@dataclass
class TicketSearchResult:
    """Result of searching for matching tickets"""
    success: bool
    matches: List[TicketMatch] = field(default_factory=list)
    best_match: Optional[TicketMatch] = None
    staff_user_id: Optional[int] = None
    error: Optional[str] = None


class EWRCentralAgent:
    """
    SQL Agent for EWRCentral database operations.

    This agent provides methods for:
    - Looking up staff by phone extension
    - Finding tickets that match phone call data
    - Running custom queries via the SQL generator

    Usage:
        agent = EWRCentralAgent()
        staff = await agent.lookup_staff_by_extension("731")
        if staff.found:
            print(f"Staff: {staff.full_name}")
    """

    _instance: Optional["EWRCentralAgent"] = None

    def __init__(self, config: Optional[Dict[str, str]] = None):
        self.config = config or EWRCENTRAL_CONFIG
        self._sql_generator = None

    @classmethod
    def get_instance(cls, config: Optional[Dict[str, str]] = None) -> "EWRCentralAgent":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance

    def _get_connection(self):
        """Get a database connection."""
        import pymssql
        return pymssql.connect(
            server=self.config["server"],
            database=self.config["database"],
            user=self.config["user"],
            password=self.config["password"],
            port=self.config["port"]
        )

    async def _execute_query(
        self,
        query: str,
        params: Optional[Tuple] = None,
        fetch_one: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Execute a SQL query and return results.

        Runs the synchronous pymssql call in a thread pool
        to avoid blocking the event loop.
        """
        def _run_query():
            conn = self._get_connection()
            try:
                cursor = conn.cursor(as_dict=True)
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)

                if fetch_one:
                    result = cursor.fetchone()
                    return [result] if result else []
                else:
                    return cursor.fetchall()
            finally:
                conn.close()

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _run_query)

    async def lookup_staff_by_extension(self, extension: str) -> StaffInfo:
        """
        Look up a staff member by their phone extension.

        Args:
            extension: The phone extension to search for

        Returns:
            StaffInfo with the staff member's details, or found=False if not found
        """
        query = """
            SELECT TOP 1
                cu.CentralUserID,
                cu.FirstName,
                cu.LastName,
                cu.FirstName + ' ' + cu.LastName AS FullName,
                cu.OfficeEmailAddress AS Email,
                LTRIM(RTRIM(cu.OfficePhoneExtension)) AS PhoneExtension
            FROM CentralUsers cu
            WHERE cu.IsActive = 1
              AND (
                  LTRIM(RTRIM(cu.OfficePhoneExtension)) = %s
                  OR LTRIM(RTRIM(cu.OfficePhoneExtension)) = %s
              )
            ORDER BY cu.LastUpdateUTC DESC
        """

        # Try extension as-is and with leading zeros stripped
        ext_stripped = extension.lstrip('0') if extension else extension

        try:
            results = await self._execute_query(query, (extension, ext_stripped), fetch_one=True)

            if results:
                row = results[0]
                return StaffInfo(
                    user_id=row['CentralUserID'],
                    first_name=row['FirstName'],
                    last_name=row['LastName'],
                    full_name=row['FullName'],
                    email=row.get('Email'),
                    extension=row.get('PhoneExtension'),
                    found=True
                )
            else:
                return StaffInfo(
                    user_id=0,
                    first_name="",
                    last_name="",
                    full_name="",
                    found=False
                )

        except Exception as e:
            logger.error(f"Failed to lookup staff by extension {extension}: {e}")
            return StaffInfo(
                user_id=0,
                first_name="",
                last_name="",
                full_name="",
                found=False
            )

    async def get_staff_user_id(self, extension: str) -> Optional[int]:
        """Get just the CentralUserID for a given extension."""
        query = """
            SELECT CentralUserID
            FROM CentralUsers
            WHERE LTRIM(RTRIM(OfficePhoneExtension)) = %s AND IsActive = 1
        """

        try:
            results = await self._execute_query(query, (extension,), fetch_one=True)
            if results:
                return results[0]['CentralUserID']
        except Exception as e:
            logger.error(f"Failed to get staff user ID: {e}")

        return None

    async def lookup_customer_by_phone(self, phone_number: str) -> CustomerInfo:
        """
        Look up customer information by phone number.

        Searches CentralTickets for the most recent ticket with matching phone number
        and extracts customer and company information.

        Args:
            phone_number: The phone number to search for (any format)

        Returns:
            CustomerInfo with customer details, or found=False if not found
        """
        if not phone_number:
            return CustomerInfo(found=False)

        # Normalize phone number - extract just digits
        phone_digits = re.sub(r'\D', '', phone_number)

        # Need at least 7 digits for a reasonable match
        if len(phone_digits) < 7:
            logger.warning(f"Phone number too short for lookup: {phone_number}")
            return CustomerInfo(found=False)

        # Use last 10 digits (or all if less than 10)
        search_digits = phone_digits[-10:] if len(phone_digits) >= 10 else phone_digits

        query = """
            SELECT TOP 1
                ct.CustomerContactName,
                ct.CustomerContactPhoneNumber,
                cc.CentralCompanyID,
                cc.CompanyName,
                cc.OfficeEmailAddress,
                ct.AddTicketDate,
                (SELECT COUNT(*) FROM CentralTickets ct2
                 WHERE REPLACE(REPLACE(REPLACE(REPLACE(ct2.CustomerContactPhoneNumber, '(', ''), ')', ''), '-', ''), ' ', '')
                 LIKE %s) AS TicketCount
            FROM CentralTickets ct
            LEFT JOIN CentralCompanies cc ON ct.CentralCompanyID = cc.CentralCompanyID
            WHERE REPLACE(REPLACE(REPLACE(REPLACE(ct.CustomerContactPhoneNumber, '(', ''), ')', ''), '-', ''), ' ', '')
                  LIKE %s
            ORDER BY ct.AddTicketDate DESC
        """

        search_pattern = f'%{search_digits}'

        try:
            results = await self._execute_query(
                query,
                (search_pattern, search_pattern),
                fetch_one=True
            )

            if results:
                row = results[0]
                return CustomerInfo(
                    customer_name=row.get('CustomerContactName'),
                    company_name=row.get('CompanyName'),
                    company_id=row.get('CentralCompanyID'),
                    phone_number=row.get('CustomerContactPhoneNumber'),
                    email=row.get('OfficeEmailAddress'),
                    last_ticket_date=row.get('AddTicketDate'),
                    ticket_count=row.get('TicketCount', 0),
                    found=True,
                    match_source="ticket"
                )
            else:
                logger.info(f"No customer found for phone: {phone_number}")
                return CustomerInfo(found=False)

        except Exception as e:
            logger.error(f"Failed to lookup customer by phone {phone_number}: {e}")
            return CustomerInfo(found=False)

    async def find_matching_tickets(
        self,
        extension: Optional[str] = None,
        phone_number: Optional[str] = None,
        call_datetime: Optional[str] = None,
        subject_keywords: Optional[List[str]] = None,
        customer_name: Optional[str] = None,
        time_window_minutes: int = 60
    ) -> TicketSearchResult:
        """
        Find tickets that might correspond to a phone call.

        Uses multiple signals to match tickets:
        - Staff member who created the ticket (from extension)
        - Customer phone number
        - Time window after the call
        - Subject keywords in ticket title
        - Customer name

        Args:
            extension: Staff extension who took the call
            phone_number: Caller's phone number
            call_datetime: ISO format datetime of the call
            subject_keywords: Keywords from call subject
            customer_name: Customer name mentioned in call
            time_window_minutes: How many minutes after call to search

        Returns:
            TicketSearchResult with scored matches
        """
        try:
            # Build dynamic query conditions
            conditions = []
            params = []

            # Get staff user ID from extension
            staff_user_id = None
            if extension:
                staff_user_id = await self.get_staff_user_id(extension)
                if staff_user_id:
                    conditions.append("ct.AddCentralUserID = %s")
                    params.append(staff_user_id)

            # Parse call datetime
            call_dt = None
            if call_datetime:
                call_dt = self._parse_datetime(call_datetime)

            if call_dt:
                end_dt = call_dt + timedelta(minutes=time_window_minutes)
                conditions.append("ct.AddTicketDate >= %s AND ct.AddTicketDate <= %s")
                params.extend([call_dt, end_dt])

            # Phone number matching
            if phone_number:
                phone_digits = re.sub(r'\D', '', phone_number)
                if len(phone_digits) >= 7:
                    conditions.append("""
                        (REPLACE(REPLACE(REPLACE(REPLACE(ct.CustomerContactPhoneNumber, '(', ''), ')', ''), '-', ''), ' ', '')
                         LIKE %s)
                    """)
                    params.append(f'%{phone_digits[-10:]}')

            # Build query
            where_clause = " AND ".join(conditions) if conditions else "1=1"

            query = f"""
                SELECT TOP 10
                    ct.CentralTicketID,
                    ct.TicketTitle,
                    ct.Note,
                    ct.AddTicketDate,
                    ct.CustomerContactName,
                    ct.CustomerContactPhoneNumber,
                    cc.CompanyName,
                    cu.FirstName + ' ' + cu.LastName AS CreatedBy,
                    ts.Description AS TicketStatus
                FROM CentralTickets ct
                LEFT JOIN CentralCompanies cc ON ct.CentralCompanyID = cc.CentralCompanyID
                LEFT JOIN CentralUsers cu ON ct.AddCentralUserID = cu.CentralUserID
                LEFT JOIN Types ts ON ct.TicketStatusTypeID = ts.TypeID
                WHERE {where_clause}
                ORDER BY ct.AddTicketDate DESC
            """

            tickets = await self._execute_query(query, tuple(params) if params else None)

            # Score and rank tickets
            scored_matches = []
            for ticket in tickets:
                match = self._score_ticket(
                    ticket,
                    staff_user_id=staff_user_id,
                    phone_number=phone_number,
                    call_dt=call_dt,
                    subject_keywords=subject_keywords,
                    customer_name=customer_name
                )
                scored_matches.append(match)

            # Sort by score
            scored_matches.sort(key=lambda x: x.match_score, reverse=True)

            # Determine best match
            best_match = None
            if scored_matches and scored_matches[0].match_score >= 50:
                best_match = scored_matches[0]

            return TicketSearchResult(
                success=True,
                matches=scored_matches,
                best_match=best_match,
                staff_user_id=staff_user_id
            )

        except Exception as e:
            logger.error(f"Failed to find matching tickets: {e}")
            return TicketSearchResult(
                success=False,
                error=str(e)
            )

    def _parse_datetime(self, dt_string: str) -> Optional[datetime]:
        """Parse various datetime formats."""
        try:
            return datetime.fromisoformat(dt_string.replace('Z', '+00:00'))
        except:
            pass

        try:
            return datetime.strptime(dt_string, "%Y%m%d-%H%M%S")
        except:
            pass

        try:
            return datetime.strptime(dt_string, "%Y-%m-%dT%H:%M:%S")
        except:
            pass

        return None

    def _score_ticket(
        self,
        ticket: Dict[str, Any],
        staff_user_id: Optional[int],
        phone_number: Optional[str],
        call_dt: Optional[datetime],
        subject_keywords: Optional[List[str]],
        customer_name: Optional[str]
    ) -> TicketMatch:
        """Score a ticket based on how well it matches the call data."""
        score = 0
        reasons = []

        # Phone number match (strongest signal - 50 points)
        if phone_number and ticket.get('CustomerContactPhoneNumber'):
            phone_digits = re.sub(r'\D', '', phone_number)
            ticket_phone = re.sub(r'\D', '', ticket['CustomerContactPhoneNumber'] or '')
            if len(phone_digits) >= 7 and len(ticket_phone) >= 7:
                if phone_digits[-7:] == ticket_phone[-7:]:
                    score += 50
                    reasons.append("phone_number_match")

        # Staff match (20 points)
        if staff_user_id:
            score += 20
            reasons.append("staff_match")

        # Time window match
        if call_dt and ticket.get('AddTicketDate'):
            ticket_dt = ticket['AddTicketDate']
            if hasattr(ticket_dt, 'replace'):
                try:
                    minutes_after = (ticket_dt.replace(tzinfo=None) - call_dt.replace(tzinfo=None)).total_seconds() / 60
                    if 0 <= minutes_after <= 30:
                        score += 30
                        reasons.append("created_within_30min")
                    elif 0 <= minutes_after <= 60:
                        score += 15
                        reasons.append("created_within_60min")
                except:
                    pass

        # Subject keyword match
        if subject_keywords and ticket.get('TicketTitle'):
            title_lower = ticket['TicketTitle'].lower()
            for keyword in subject_keywords:
                if keyword.lower() in title_lower:
                    score += 10
                    reasons.append(f"keyword_match:{keyword}")

        # Customer name match
        if customer_name and ticket.get('CustomerContactName'):
            if customer_name.lower() in ticket['CustomerContactName'].lower():
                score += 25
                reasons.append("customer_name_match")

        # Convert datetime for the result
        created_date = ticket.get('AddTicketDate')
        if created_date and not isinstance(created_date, str):
            created_date = created_date

        return TicketMatch(
            ticket_id=ticket['CentralTicketID'],
            title=ticket.get('TicketTitle', ''),
            note=ticket.get('Note'),
            created_date=created_date,
            customer_name=ticket.get('CustomerContactName'),
            customer_phone=ticket.get('CustomerContactPhoneNumber'),
            company_name=ticket.get('CompanyName'),
            created_by=ticket.get('CreatedBy', 'Unknown'),
            status=ticket.get('TicketStatus'),
            match_score=score,
            match_reasons=reasons
        )

    async def execute_natural_language_query(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a natural language query using the SQL generator.

        This method uses the LLM-based SQL generator to convert
        a natural language question into SQL and execute it.

        Args:
            question: Natural language question about EWRCentral data
            context: Optional context for the query

        Returns:
            Dict with success, rows, columns, and other metadata
        """
        from sql_pipeline import get_query_pipeline
        from api.sql_routes_new import execute_sql_query, add_safety_limit

        try:
            # Get the SQL Query Pipeline
            pipeline = await get_query_pipeline()

            # Generate SQL from natural language
            gen_result = await pipeline.generate(
                question=question,
                database="EWRCentral",
                server=self.config["server"],
            )

            if not gen_result.success:
                return {
                    "success": False,
                    "error": gen_result.error,
                    "security_blocked": gen_result.security_blocked
                }

            # Add safety limit
            sql = add_safety_limit(gen_result.sql)

            # Execute the query
            result = execute_sql_query(
                sql=sql,
                server=self.config["server"],
                database=self.config["database"],
                user=self.config["user"],
                password=self.config["password"],
                auth_type="windows"
            )

            return {
                "success": result.get("success", False),
                "sql": sql,
                "columns": result.get("columns", []),
                "rows": result.get("rows", []),
                "row_count": result.get("rowCount", 0),
                "error": result.get("error"),
                "is_exact_match": gen_result.is_exact_match,
                "model_used": "qwen2.5-coder"
            }

        except Exception as e:
            logger.error(f"Natural language query failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }


# Singleton accessor
_agent_instance: Optional[EWRCentralAgent] = None


def get_ewrcentral_agent() -> EWRCentralAgent:
    """Get the singleton EWRCentral agent instance."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = EWRCentralAgent()
    return _agent_instance


async def get_ewrcentral_agent_async() -> EWRCentralAgent:
    """Async accessor for the EWRCentral agent."""
    return get_ewrcentral_agent()
