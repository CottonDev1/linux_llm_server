"""
MongoDB repository for LLM trace storage and retrieval.

Handles all database operations for:
- Storing traces
- Querying traces with filters
- Aggregating statistics
- Real-time metrics
"""
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from pymongo import MongoClient, DESCENDING, ASCENDING
from pymongo.collection import Collection
from pymongo.database import Database

from .models import (
    LLMTrace, TraceFilter, TraceStats, RealtimeMetrics,
    TraceStatus, Pipeline
)


class LLMTraceRepository:
    """
    MongoDB repository for LLM traces.
    
    Collection: llm_traces
    Indexes:
    - timestamp (descending) - for time-based queries
    - pipeline + timestamp - for pipeline filtering
    - trace_id (unique) - for lookups
    - outcome.status + timestamp - for error queries
    - context.user_id + timestamp - for user queries
    """
    
    COLLECTION_NAME = "llm_traces"
    
    def __init__(self, mongodb_uri: str, database_name: str = "llm_website"):
        """Initialize repository with MongoDB connection."""
        self.client = MongoClient(mongodb_uri)
        self.db: Database = self.client[database_name]
        self.collection: Collection = self.db[self.COLLECTION_NAME]
        self._ensure_indexes()
    
    def _ensure_indexes(self):
        """Create indexes for efficient querying."""
        # Time-based queries (most common)
        self.collection.create_index(
            [("timestamp", DESCENDING)],
            name="idx_timestamp"
        )
        
        # Pipeline + time filtering
        self.collection.create_index(
            [("pipeline", ASCENDING), ("timestamp", DESCENDING)],
            name="idx_pipeline_timestamp"
        )
        
        # Unique trace lookup
        self.collection.create_index(
            [("trace_id", ASCENDING)],
            unique=True,
            name="idx_trace_id"
        )
        
        # Error queries
        self.collection.create_index(
            [("outcome.status", ASCENDING), ("timestamp", DESCENDING)],
            name="idx_status_timestamp"
        )
        
        # User queries
        self.collection.create_index(
            [("context.user_id", ASCENDING), ("timestamp", DESCENDING)],
            name="idx_user_timestamp",
            sparse=True
        )
        
        # Session queries
        self.collection.create_index(
            [("context.session_id", ASCENDING), ("timestamp", DESCENDING)],
            name="idx_session_timestamp",
            sparse=True
        )
        
        # Performance queries (slow requests)
        self.collection.create_index(
            [("total_duration_ms", DESCENDING)],
            name="idx_duration"
        )
    
    async def insert_trace(self, trace: LLMTrace) -> str:
        """Insert a new trace document."""
        doc = trace.model_dump()
        self.collection.insert_one(doc)
        return trace.trace_id
    
    def insert_trace_sync(self, trace: LLMTrace) -> str:
        """Synchronous version of insert_trace."""
        doc = trace.model_dump()
        self.collection.insert_one(doc)
        return trace.trace_id
    
    async def get_trace(self, trace_id: str) -> Optional[LLMTrace]:
        """Get a single trace by ID."""
        doc = self.collection.find_one({"trace_id": trace_id})
        if doc:
            doc.pop("_id", None)
            return LLMTrace(**doc)
        return None
    
    def get_trace_sync(self, trace_id: str) -> Optional[LLMTrace]:
        """Synchronous version of get_trace."""
        doc = self.collection.find_one({"trace_id": trace_id})
        if doc:
            doc.pop("_id", None)
            return LLMTrace(**doc)
        return None
    
    async def query_traces(self, filter: TraceFilter) -> List[LLMTrace]:
        """Query traces with filters."""
        query = self._build_query(filter)
        
        sort_direction = DESCENDING if filter.sort_order == "desc" else ASCENDING
        
        cursor = self.collection.find(query).sort(
            filter.sort_by, sort_direction
        ).skip(filter.skip).limit(filter.limit)
        
        traces = []
        for doc in cursor:
            doc.pop("_id", None)
            traces.append(LLMTrace(**doc))
        
        return traces
    
    def query_traces_sync(self, filter: TraceFilter) -> List[LLMTrace]:
        """Synchronous version of query_traces."""
        query = self._build_query(filter)
        
        sort_direction = DESCENDING if filter.sort_order == "desc" else ASCENDING
        
        cursor = self.collection.find(query).sort(
            filter.sort_by, sort_direction
        ).skip(filter.skip).limit(filter.limit)
        
        traces = []
        for doc in cursor:
            doc.pop("_id", None)
            traces.append(LLMTrace(**doc))
        
        return traces
    
    async def count_traces(self, filter: TraceFilter) -> int:
        """Count traces matching filter."""
        query = self._build_query(filter)
        return self.collection.count_documents(query)
    
    def _build_query(self, filter: TraceFilter) -> Dict[str, Any]:
        """Build MongoDB query from filter."""
        query = {}
        
        if filter.pipeline:
            query["pipeline"] = filter.pipeline.value if isinstance(filter.pipeline, Pipeline) else filter.pipeline
        
        if filter.status:
            query["outcome.status"] = filter.status.value if isinstance(filter.status, TraceStatus) else filter.status
        
        if filter.user_id:
            query["context.user_id"] = filter.user_id
        
        if filter.session_id:
            query["context.session_id"] = filter.session_id
        
        if filter.operation:
            query["operation"] = filter.operation
        
        # Time range
        if filter.start_time or filter.end_time:
            query["timestamp"] = {}
            if filter.start_time:
                query["timestamp"]["$gte"] = filter.start_time
            if filter.end_time:
                query["timestamp"]["$lte"] = filter.end_time
        
        # Performance filters
        if filter.min_duration_ms is not None:
            query["total_duration_ms"] = {"$gte": filter.min_duration_ms}
        if filter.max_duration_ms is not None:
            if "total_duration_ms" in query:
                query["total_duration_ms"]["$lte"] = filter.max_duration_ms
            else:
                query["total_duration_ms"] = {"$lte": filter.max_duration_ms}
        
        if filter.min_tokens is not None:
            query["$expr"] = {
                "$gte": [
                    {"$add": [
                        {"$ifNull": ["$response.tokens_evaluated", 0]},
                        {"$ifNull": ["$response.tokens_predicted", 0]}
                    ]},
                    filter.min_tokens
                ]
            }
        
        return query
    
    async def get_stats(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        pipeline: Optional[Pipeline] = None,
    ) -> TraceStats:
        """Get aggregated statistics for traces."""
        
        match_stage: Dict[str, Any] = {}
        
        if start_time or end_time:
            match_stage["timestamp"] = {}
            if start_time:
                match_stage["timestamp"]["$gte"] = start_time
            if end_time:
                match_stage["timestamp"]["$lte"] = end_time
        
        if pipeline:
            match_stage["pipeline"] = pipeline.value
        
        pipeline_stages = []
        
        if match_stage:
            pipeline_stages.append({"$match": match_stage})
        
        pipeline_stages.extend([
            {
                "$group": {
                    "_id": None,
                    "total_traces": {"$sum": 1},
                    "success_count": {
                        "$sum": {"$cond": [{"$eq": ["$outcome.status", "success"]}, 1, 0]}
                    },
                    "error_count": {
                        "$sum": {"$cond": [{"$eq": ["$outcome.status", "error"]}, 1, 0]}
                    },
                    "total_prompt_tokens": {"$sum": {"$ifNull": ["$response.tokens_evaluated", 0]}},
                    "total_completion_tokens": {"$sum": {"$ifNull": ["$response.tokens_predicted", 0]}},
                    "avg_prompt_tokens": {"$avg": {"$ifNull": ["$response.tokens_evaluated", 0]}},
                    "avg_completion_tokens": {"$avg": {"$ifNull": ["$response.tokens_predicted", 0]}},
                    "avg_duration_ms": {"$avg": "$total_duration_ms"},
                    "min_duration_ms": {"$min": "$total_duration_ms"},
                    "max_duration_ms": {"$max": "$total_duration_ms"},
                    "durations": {"$push": "$total_duration_ms"},
                    "tokens_per_second": {
                        "$avg": {
                            "$cond": [
                                {"$gt": ["$total_duration_ms", 0]},
                                {"$divide": [
                                    {"$ifNull": ["$response.tokens_predicted", 0]},
                                    {"$divide": ["$total_duration_ms", 1000]}
                                ]},
                                0
                            ]
                        }
                    },
                }
            }
        ])
        
        result = list(self.collection.aggregate(pipeline_stages))
        
        if not result:
            return TraceStats(period_start=start_time, period_end=end_time)
        
        data = result[0]
        
        # Calculate percentiles from durations
        durations = sorted([d for d in data.get("durations", []) if d is not None])
        p50 = durations[len(durations) // 2] if durations else 0
        p95 = durations[int(len(durations) * 0.95)] if durations else 0
        p99 = durations[int(len(durations) * 0.99)] if durations else 0
        
        # Get breakdown by pipeline
        by_pipeline = {}
        pipeline_breakdown = self.collection.aggregate([
            {"$match": match_stage} if match_stage else {"$match": {}},
            {"$group": {"_id": "$pipeline", "count": {"$sum": 1}}}
        ])
        for item in pipeline_breakdown:
            if item["_id"]:
                by_pipeline[item["_id"]] = item["count"]
        
        # Get breakdown by status
        by_status = {}
        status_breakdown = self.collection.aggregate([
            {"$match": match_stage} if match_stage else {"$match": {}},
            {"$group": {"_id": "$outcome.status", "count": {"$sum": 1}}}
        ])
        for item in status_breakdown:
            if item["_id"]:
                by_status[item["_id"]] = item["count"]
        
        return TraceStats(
            total_traces=data.get("total_traces", 0),
            success_count=data.get("success_count", 0),
            error_count=data.get("error_count", 0),
            total_prompt_tokens=data.get("total_prompt_tokens", 0),
            total_completion_tokens=data.get("total_completion_tokens", 0),
            avg_prompt_tokens=data.get("avg_prompt_tokens", 0) or 0,
            avg_completion_tokens=data.get("avg_completion_tokens", 0) or 0,
            avg_duration_ms=data.get("avg_duration_ms", 0) or 0,
            min_duration_ms=data.get("min_duration_ms", 0) or 0,
            max_duration_ms=data.get("max_duration_ms", 0) or 0,
            p50_duration_ms=p50,
            p95_duration_ms=p95,
            p99_duration_ms=p99,
            avg_tokens_per_second=data.get("tokens_per_second", 0) or 0,
            by_pipeline=by_pipeline,
            by_status=by_status,
            period_start=start_time,
            period_end=end_time,
        )
    
    async def get_realtime_metrics(self) -> RealtimeMetrics:
        """Get real-time metrics for dashboard."""
        now = datetime.utcnow()
        one_minute_ago = now - timedelta(minutes=1)
        one_hour_ago = now - timedelta(hours=1)
        
        # Requests in last minute
        requests_last_minute = self.collection.count_documents({
            "timestamp": {"$gte": one_minute_ago}
        })
        
        # Requests in last hour
        requests_last_hour = self.collection.count_documents({
            "timestamp": {"$gte": one_hour_ago}
        })
        
        # Error rate last hour
        errors_last_hour = self.collection.count_documents({
            "timestamp": {"$gte": one_hour_ago},
            "outcome.status": "error"
        })
        error_rate = (errors_last_hour / requests_last_hour * 100) if requests_last_hour > 0 else 0
        
        # Average latency last hour
        latency_result = list(self.collection.aggregate([
            {"$match": {"timestamp": {"$gte": one_hour_ago}}},
            {"$group": {"_id": None, "avg": {"$avg": "$total_duration_ms"}}}
        ]))
        avg_latency = latency_result[0]["avg"] if latency_result and latency_result[0].get("avg") else 0
        
        # Total tokens last hour
        tokens_result = list(self.collection.aggregate([
            {"$match": {"timestamp": {"$gte": one_hour_ago}}},
            {"$group": {
                "_id": None,
                "total": {"$sum": {
                    "$add": [
                        {"$ifNull": ["$response.tokens_evaluated", 0]},
                        {"$ifNull": ["$response.tokens_predicted", 0]}
                    ]
                }}
            }}
        ]))
        total_tokens = tokens_result[0]["total"] if tokens_result and tokens_result[0].get("total") else 0
        
        # Recent errors (last 10)
        recent_errors = list(self.collection.find(
            {"outcome.status": "error"},
            {"trace_id": 1, "timestamp": 1, "pipeline": 1, "outcome.error_message": 1, "_id": 0}
        ).sort("timestamp", DESCENDING).limit(10))
        
        return RealtimeMetrics(
            active_requests=0,  # Would need separate tracking
            requests_last_minute=requests_last_minute,
            requests_last_hour=requests_last_hour,
            error_rate_last_hour=error_rate,
            avg_latency_ms=avg_latency,
            total_tokens_last_hour=total_tokens,
            recent_errors=recent_errors,
            updated_at=now,
        )
    
    async def get_latency_timeline(
        self,
        start_time: datetime,
        end_time: datetime,
        interval_minutes: int = 5,
        pipeline: Optional[Pipeline] = None,
    ) -> List[Dict[str, Any]]:
        """Get latency over time for charts."""
        match_stage: Dict[str, Any] = {
            "timestamp": {"$gte": start_time, "$lte": end_time}
        }
        if pipeline:
            match_stage["pipeline"] = pipeline.value
        
        result = list(self.collection.aggregate([
            {"$match": match_stage},
            {
                "$group": {
                    "_id": {
                        "$toDate": {
                            "$subtract": [
                                {"$toLong": "$timestamp"},
                                {"$mod": [{"$toLong": "$timestamp"}, interval_minutes * 60 * 1000]}
                            ]
                        }
                    },
                    "avg_latency": {"$avg": "$total_duration_ms"},
                    "min_latency": {"$min": "$total_duration_ms"},
                    "max_latency": {"$max": "$total_duration_ms"},
                    "count": {"$sum": 1}
                }
            },
            {"$sort": {"_id": 1}}
        ]))
        
        return [
            {
                "timestamp": item["_id"].isoformat() if item["_id"] else None,
                "avg_latency": item["avg_latency"] or 0,
                "min_latency": item["min_latency"] or 0,
                "max_latency": item["max_latency"] or 0,
                "count": item["count"]
            }
            for item in result
        ]
    
    async def get_token_usage_timeline(
        self,
        start_time: datetime,
        end_time: datetime,
        interval_minutes: int = 60,
    ) -> List[Dict[str, Any]]:
        """Get token usage over time for charts."""
        result = list(self.collection.aggregate([
            {"$match": {"timestamp": {"$gte": start_time, "$lte": end_time}}},
            {
                "$group": {
                    "_id": {
                        "$toDate": {
                            "$subtract": [
                                {"$toLong": "$timestamp"},
                                {"$mod": [{"$toLong": "$timestamp"}, interval_minutes * 60 * 1000]}
                            ]
                        }
                    },
                    "prompt_tokens": {"$sum": {"$ifNull": ["$response.tokens_evaluated", 0]}},
                    "completion_tokens": {"$sum": {"$ifNull": ["$response.tokens_predicted", 0]}},
                    "count": {"$sum": 1}
                }
            },
            {"$sort": {"_id": 1}}
        ]))
        
        return [
            {
                "timestamp": item["_id"].isoformat() if item["_id"] else None,
                "prompt_tokens": item["prompt_tokens"],
                "completion_tokens": item["completion_tokens"],
                "total_tokens": item["prompt_tokens"] + item["completion_tokens"],
                "count": item["count"]
            }
            for item in result
        ]
    
    async def delete_old_traces(self, days_to_keep: int = 30) -> int:
        """Delete traces older than specified days."""
        cutoff = datetime.utcnow() - timedelta(days=days_to_keep)
        result = self.collection.delete_many({"timestamp": {"$lt": cutoff}})
        return result.deleted_count
    
    def update_trace_outcome(
        self,
        trace_id: str,
        validation_passed: Optional[bool] = None,
        executed: Optional[bool] = None,
        rows_returned: Optional[int] = None,
        user_feedback: Optional[str] = None,
    ) -> bool:
        """Update trace with outcome information."""
        update = {}
        if validation_passed is not None:
            update["outcome.validation_passed"] = validation_passed
        if executed is not None:
            update["outcome.executed"] = executed
        if rows_returned is not None:
            update["outcome.rows_returned"] = rows_returned
        if user_feedback is not None:
            update["outcome.user_feedback"] = user_feedback
        
        if update:
            result = self.collection.update_one(
                {"trace_id": trace_id},
                {"$set": update}
            )
            return result.modified_count > 0
        return False
