"""
Prefect Deployment Configuration for Autonomous SQL Training
=============================================================

This file configures the Prefect deployment for the autonomous SQL training
pipeline that runs overnight (7 PM - 7 AM CST).

Usage:
    python deployment.py           # Register deployments
    prefect deployment run autonomous-sql-training/nightly
"""

from datetime import timedelta
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule

from autonomous_sql_training_flow import autonomous_sql_training_flow


def create_nightly_deployment():
    """Create deployment for nightly autonomous training."""

    deployment = Deployment.build_from_flow(
        flow=autonomous_sql_training_flow,
        name="nightly",
        description="Nightly autonomous SQL training (7 PM - 7 AM CST)",
        version="1.0",
        tags=["sql-training", "autonomous", "nightly"],
        parameters={
            "database": "EWRCentral",
            "host": "CHAD-PC",
            "max_tables": 50,
            "batch_size": 5,
            "questions_per_table": 3,
            "cutoff_hour": 7,
            "cutoff_tz": "America/Chicago"
        },
        schedule=CronSchedule(
            cron="0 19 * * *",  # 7:00 PM daily
            timezone="America/Chicago"
        ),
        work_queue_name="sql-training",
        infra_overrides={
            "env": {
                "PREFECT_LOGGING_LEVEL": "INFO"
            }
        }
    )

    return deployment


def create_test_deployment():
    """Create deployment for testing with reduced scope."""

    deployment = Deployment.build_from_flow(
        flow=autonomous_sql_training_flow,
        name="test",
        description="Test deployment with reduced scope",
        version="1.0",
        tags=["sql-training", "test"],
        parameters={
            "database": "EWRCentral",
            "host": "CHAD-PC",
            "max_tables": 3,
            "batch_size": 2,
            "questions_per_table": 2,
            "cutoff_hour": 23,  # Late night cutoff for testing
            "cutoff_tz": "America/Chicago"
        },
        work_queue_name="default"
    )

    return deployment


def create_manual_deployment():
    """Create deployment for manual triggering."""

    deployment = Deployment.build_from_flow(
        flow=autonomous_sql_training_flow,
        name="manual",
        description="Manual deployment for on-demand training",
        version="1.0",
        tags=["sql-training", "manual"],
        parameters={
            "database": "EWRCentral",
            "host": "CHAD-PC",
            "max_tables": 25,
            "batch_size": 5,
            "questions_per_table": 3,
            "cutoff_hour": 23,
            "cutoff_tz": "America/Chicago"
        },
        work_queue_name="sql-training"
    )

    return deployment


if __name__ == "__main__":
    import asyncio

    async def deploy_all():
        """Deploy all configurations."""
        print("Creating deployments...")

        # Create nightly deployment
        nightly = create_nightly_deployment()
        nightly_id = await nightly.apply()
        print(f"✓ Nightly deployment created: {nightly_id}")

        # Create test deployment
        test = create_test_deployment()
        test_id = await test.apply()
        print(f"✓ Test deployment created: {test_id}")

        # Create manual deployment
        manual = create_manual_deployment()
        manual_id = await manual.apply()
        print(f"✓ Manual deployment created: {manual_id}")

        print("\nDeployments ready!")
        print("Run with: prefect deployment run autonomous-sql-training/nightly")
        print("Test with: prefect deployment run autonomous-sql-training/test")

    asyncio.run(deploy_all())
