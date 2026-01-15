"""
Database Service

Handles database lookups for staff and customer information.
"""

import re
from typing import Dict, Optional


class DatabaseService:
    """
    Service for database lookups related to audio analysis.

    Handles:
    - Staff lookup by phone extension
    - Customer lookup by phone number
    """

    _instance: Optional['DatabaseService'] = None

    @classmethod
    def get_instance(cls) -> 'DatabaseService':
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def lookup_staff_by_extension(self, extension: str) -> Optional[str]:
        """
        Look up staff name from EWRCentral database by phone extension.

        Args:
            extension: Phone extension to look up

        Returns:
            Staff full name if found, None otherwise
        """
        if not extension:
            return None

        try:
            import pymssql

            connection = pymssql.connect(
                server='EWRSQLPROD',
                database='EWRCentral',
                user='EWR\\chad.walker',
                password='6454@@Christina',
                port='1433',
                timeout=10
            )
            cursor = connection.cursor(as_dict=True)

            # Query for staff by extension
            ext_stripped = extension.lstrip('0') if extension else extension
            cursor.execute("""
                SELECT TOP 1
                    FirstName + ' ' + LastName AS FullName
                FROM CentralUsers
                WHERE IsActive = 1
                  AND (
                      LTRIM(RTRIM(OfficePhoneExtension)) = %s
                      OR LTRIM(RTRIM(OfficePhoneExtension)) = %s
                  )
                ORDER BY LastUpdateUTC DESC
            """, (extension, ext_stripped))

            result = cursor.fetchone()
            cursor.close()
            connection.close()

            if result:
                return result['FullName']
            return None

        except Exception as e:
            print(f"Staff lookup error: {e}")
            return None

    async def lookup_customer_by_phone(self, phone_number: str) -> Optional[Dict]:
        """
        Look up customer information from EWRCentral database by phone number.

        TEMPORARILY DISABLED: OfficeEmailAddress column doesn't exist in database.
        TODO: Fix the SQL query to use correct column names.
        """
        # DISABLED - OfficeEmailAddress column error
        # return {"found": False} immediately to skip the lookup
        print("Customer lookup disabled - OfficeEmailAddress column error")
        return {"found": False}

        # --- ORIGINAL CODE COMMENTED OUT ---
        # if not phone_number:
        #     return None
        #
        # # Normalize phone number - extract just digits
        # phone_digits = re.sub(r'\D', '', phone_number)
        #
        # # Need at least 7 digits for a reasonable match
        # if len(phone_digits) < 7:
        #     print(f"Phone number too short for lookup: {phone_number}")
        #     return {"found": False}
        #
        # # Use last 10 digits (or all if less than 10)
        # search_digits = phone_digits[-10:] if len(phone_digits) >= 10 else phone_digits
        # search_pattern = f'%{search_digits}'
        #
        # try:
        #     import pymssql
        #
        #     connection = pymssql.connect(
        #         server='EWRSQLPROD',
        #         database='EWRCentral',
        #         user='EWR\\chad.walker',
        #         password='6454@@Christina',
        #         port='1433',
        #         timeout=10
        #     )
        #     cursor = connection.cursor(as_dict=True)
        #
        #     cursor.execute("""
        #         SELECT TOP 1
        #             ct.CustomerContactName,
        #             ct.CustomerContactPhoneNumber,
        #             cc.CentralCompanyID,
        #             cc.CompanyName,
        #             cc.OfficeEmailAddress,
        #             ct.AddTicketDate,
        #             (SELECT COUNT(*) FROM CentralTickets ct2
        #              WHERE REPLACE(REPLACE(REPLACE(REPLACE(ct2.CustomerContactPhoneNumber, '(', ''), ')', ''), '-', ''), ' ', '')
        #              LIKE %s) AS TicketCount
        #         FROM CentralTickets ct
        #         LEFT JOIN CentralCompanies cc ON ct.CentralCompanyID = cc.CentralCompanyID
        #         WHERE REPLACE(REPLACE(REPLACE(REPLACE(ct.CustomerContactPhoneNumber, '(', ''), ')', ''), '-', ''), ' ', '')
        #               LIKE %s
        #         ORDER BY ct.AddTicketDate DESC
        #     """, (search_pattern, search_pattern))
        #
        #     result = cursor.fetchone()
        #     cursor.close()
        #     connection.close()
        #
        #     if result:
        #         return {
        #             "found": True,
        #             "customer_name": result.get('CustomerContactName'),
        #             "company_name": result.get('CompanyName'),
        #             "company_id": result.get('CentralCompanyID'),
        #             "email": result.get('OfficeEmailAddress'),
        #             "ticket_count": result.get('TicketCount', 0),
        #             "match_source": "ticket"
        #         }
        #
        #     return {"found": False}
        #
        # except Exception as e:
        #     print(f"Customer lookup error: {e}")
        #     return {"found": False}


def get_database_service() -> DatabaseService:
    """Get the singleton database service instance"""
    return DatabaseService.get_instance()
