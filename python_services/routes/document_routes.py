"""Document routes for document management and search."""
from fastapi import APIRouter, Request, Query, HTTPException, UploadFile, File, Form
from typing import Optional, List
import tempfile
import os
import asyncio

from data_models import (
    DocumentCreate, DocumentUpdate,
    StoreResponse, DeleteResponse, StatsResponse
)
from mongodb import get_mongodb_service
from log_service import log_pipeline, log_error
from config import COLLECTION_DOCUMENTS

router = APIRouter(prefix="/documents", tags=["Documents"])

# Concurrency limit for parallel document processing
MAX_CONCURRENT_UPLOADS = 4


@router.post("", response_model=StoreResponse)
async def store_document(document: DocumentCreate, request: Request):
    """Store a new document with automatic chunking and embedding"""
    user_ip = request.client.host if request.client else "Unknown"

    log_pipeline("DOCUMENT", user_ip, "Storing document",
                 document.title[:100] if document.title else "Untitled",
                 details={
                     "department": document.department,
                     "type": document.type
                 })

    mongodb = get_mongodb_service()
    result = await mongodb.store_document(
        title=document.title,
        content=document.content,
        department=document.department,
        doc_type=document.type,
        subject=document.subject,
        file_name=document.file_name,
        file_size=document.file_size,
        tags=document.tags,
        metadata=document.metadata
    )

    log_pipeline("DOCUMENT", user_ip, "Document stored",
                 details={
                     "document_id": result.get("parent_id"),
                     "chunks_created": result.get("chunks_created", 0)
                 })

    return StoreResponse(**result)


@router.post("/upload")
async def upload_document(
    request: Request,
    file: UploadFile = File(..., description="Document file (PDF, DOCX, XLSX, TXT)"),
    department: str = Form("general", description="Department (Tier 1)"),
    type: str = Form("documentation", description="Document type (Tier 2)"),
    subject: Optional[str] = Form(None, description="Subject/Product (Tier 3)"),
    tags: Optional[str] = Form(None, description="Comma-separated tags")
):
    """
    Upload a document file, extract text, generate embeddings, and store.

    Supported formats: PDF, DOCX, DOC, XLSX, XLS, TXT, MD, CSV, JSON
    """
    user_ip = request.client.host if request.client else "Unknown"

    log_pipeline("DOCUMENT", user_ip, "File upload started",
                 file.filename or "unknown",
                 details={"department": department, "type": type})

    try:
        # Import document processor
        from document_processor import DocumentProcessor
        processor = DocumentProcessor()

        # Save uploaded file to temp location
        suffix = os.path.splitext(file.filename or "")[1] or ".txt"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            # Process document to extract text
            result = await processor.process_file(
                file_path=tmp_path,
                original_filename=file.filename
            )

            # ProcessedDocument is a dataclass, access attributes directly
            if not result.success:
                raise HTTPException(status_code=400, detail=result.error or "Failed to process document")

            extracted_text = result.content or ""
            # Convert DocumentMetadata dataclass to dict for storage
            metadata = {
                "title": result.metadata.title,
                "author": result.metadata.author,
                "created_date": result.metadata.created_date,
                "page_count": result.metadata.page_count,
                "word_count": result.metadata.word_count,
                "has_tables": result.metadata.has_tables,
                "has_images": result.metadata.has_images,
                "table_count": result.metadata.table_count,
                "sheet_names": result.metadata.sheet_names,
                "language": result.metadata.language,
                "file_hash": result.metadata.file_hash
            }

            if not extracted_text or len(extracted_text.strip()) < 10:
                raise HTTPException(status_code=400, detail="Document contains no extractable text")

            # Parse tags
            tag_list = [t.strip() for t in (tags or "").split(",") if t.strip()] if tags else []

            # Store document with embeddings
            mongodb = get_mongodb_service()
            store_result = await mongodb.store_document(
                title=file.filename or "Untitled",
                content=extracted_text,
                department=department,
                doc_type=type,
                subject=subject,
                file_name=file.filename,
                file_size=len(content),
                tags=tag_list,
                metadata={
                    **metadata,
                    "original_filename": file.filename,
                    "content_type": file.content_type,
                    "extraction_method": result.content_type
                }
            )

            log_pipeline("DOCUMENT", user_ip, "File upload complete",
                         file.filename or "unknown",
                         details={
                             "document_id": store_result.get("parent_id"),
                             "chunks_created": store_result.get("chunks_created", 0),
                             "text_length": len(extracted_text)
                         })

            return {
                "success": True,
                "message": f"Document uploaded and processed successfully",
                "document_id": store_result.get("parent_id"),
                "chunks_created": store_result.get("chunks_created", 0),
                "text_extracted": len(extracted_text),
                "filename": file.filename
            }

        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except HTTPException:
        raise
    except ImportError as e:
        log_error("DOCUMENT", user_ip, f"Missing dependency: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Document processing not available: {str(e)}")
    except Exception as e:
        log_error("DOCUMENT", user_ip, f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.post("/upload-batch")
async def upload_documents_batch(
    request: Request,
    files: List[UploadFile] = File(..., description="Document files (PDF, DOCX, XLSX, TXT)"),
    department: str = Form("general", description="Department (Tier 1)"),
    type: str = Form("documentation", description="Document type (Tier 2)"),
    subject: Optional[str] = Form(None, description="Subject/Product (Tier 3)"),
    tags: Optional[str] = Form(None, description="Comma-separated tags")
):
    """
    Upload multiple documents in parallel with controlled concurrency.

    Processes up to MAX_CONCURRENT_UPLOADS files simultaneously for 3-4x faster batch uploads.
    Supported formats: PDF, DOCX, DOC, XLSX, XLS, TXT, MD, CSV, JSON
    """
    user_ip = request.client.host if request.client else "Unknown"

    log_pipeline("DOCUMENT", user_ip, "Batch upload started",
                 f"{len(files)} files",
                 details={"department": department, "type": type})

    # Create semaphore for controlled concurrency
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_UPLOADS)

    async def process_single_file(file: UploadFile) -> dict:
        """Process a single file with semaphore-controlled concurrency."""
        async with semaphore:
            try:
                from document_processor import DocumentProcessor
                processor = DocumentProcessor()

                # Save uploaded file to temp location
                suffix = os.path.splitext(file.filename or "")[1] or ".txt"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    content = await file.read()
                    tmp.write(content)
                    tmp_path = tmp.name

                try:
                    # Process document to extract text
                    result = await processor.process_file(
                        file_path=tmp_path,
                        original_filename=file.filename
                    )

                    if not result.success:
                        return {
                            "filename": file.filename,
                            "success": False,
                            "error": result.error or "Failed to process document"
                        }

                    extracted_text = result.content or ""
                    metadata = {
                        "title": result.metadata.title,
                        "author": result.metadata.author,
                        "created_date": result.metadata.created_date,
                        "page_count": result.metadata.page_count,
                        "word_count": result.metadata.word_count,
                        "has_tables": result.metadata.has_tables,
                        "has_images": result.metadata.has_images,
                        "table_count": result.metadata.table_count,
                        "sheet_names": result.metadata.sheet_names,
                        "language": result.metadata.language,
                        "file_hash": result.metadata.file_hash
                    }

                    if not extracted_text or len(extracted_text.strip()) < 10:
                        return {
                            "filename": file.filename,
                            "success": False,
                            "error": "Document contains no extractable text"
                        }

                    # Parse tags
                    tag_list = [t.strip() for t in (tags or "").split(",") if t.strip()] if tags else []

                    # Store document with embeddings
                    mongodb = get_mongodb_service()
                    store_result = await mongodb.store_document(
                        title=file.filename or "Untitled",
                        content=extracted_text,
                        department=department,
                        doc_type=type,
                        subject=subject,
                        file_name=file.filename,
                        file_size=len(content),
                        tags=tag_list,
                        metadata={
                            **metadata,
                            "original_filename": file.filename,
                            "content_type": file.content_type,
                            "extraction_method": result.content_type
                        }
                    )

                    return {
                        "filename": file.filename,
                        "success": True,
                        "document_id": store_result.get("parent_id"),
                        "chunks_created": store_result.get("chunks_created", 0),
                        "text_extracted": len(extracted_text)
                    }

                finally:
                    # Clean up temp file
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)

            except Exception as e:
                return {
                    "filename": file.filename,
                    "success": False,
                    "error": str(e)
                }

    # Process all files in parallel with controlled concurrency
    results = await asyncio.gather(*[process_single_file(f) for f in files])

    # Summarize results
    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]

    log_pipeline("DOCUMENT", user_ip, "Batch upload complete",
                 f"{len(successful)}/{len(files)} successful",
                 details={
                     "total": len(files),
                     "successful": len(successful),
                     "failed": len(failed),
                     "total_chunks": sum(r.get("chunks_created", 0) for r in successful)
                 })

    return {
        "success": len(failed) == 0,
        "message": f"Processed {len(files)} files: {len(successful)} successful, {len(failed)} failed",
        "total_files": len(files),
        "successful_count": len(successful),
        "failed_count": len(failed),
        "total_chunks_created": sum(r.get("chunks_created", 0) for r in successful),
        "results": results
    }


@router.get("/search")
async def search_documents(
    request: Request,
    query: str = Query(..., description="Search query"),
    limit: int = Query(5, ge=1, le=50, description="Maximum results"),
    department: Optional[str] = Query(None, description="Filter by department"),
    type: Optional[str] = Query(None, description="Filter by type"),
    subject: Optional[str] = Query(None, description="Filter by subject")
):
    """Search documents using semantic similarity"""
    user_ip = request.client.host if request.client else "Unknown"

    mongodb = get_mongodb_service()
    results = await mongodb.search_documents(
        query=query,
        limit=limit,
        department=department,
        doc_type=type,
        subject=subject
    )

    log_pipeline("DOCUMENT", user_ip, "Document search",
                 query[:100] if query else "",
                 details={
                     "limit": limit,
                     "results_found": len(results),
                     "department": department,
                     "type": type,
                     "subject": subject
                 })

    return results


@router.get("/total-count")
async def get_total_document_count(request: Request):
    """Get total count of unique documents (not chunks)"""
    user_ip = request.client.host if request.client else "Unknown"

    mongodb = get_mongodb_service()
    collection = mongodb.db[COLLECTION_DOCUMENTS]

    # Count unique documents by grouping on title
    pipeline = [
        {"$group": {"_id": "$title"}},
        {"$count": "total"}
    ]

    result = await collection.aggregate(pipeline).to_list(length=1)
    total = result[0]["total"] if result else 0

    log_pipeline("DOCUMENT", user_ip, "Total document count", details={"total": total})

    return {"success": True, "total_documents": total}


@router.get("/aggregate")
async def aggregate_documents(
    request: Request,
    department: Optional[str] = Query(None, description="Filter by department"),
    doc_type: Optional[str] = Query(None, alias="type", description="Filter by document type"),
    subject: Optional[str] = Query(None, description="Filter by subject/product")
):
    """
    Get document counts and titles using MongoDB aggregation.
    Returns total count and list of unique document titles based on filters.
    """
    user_ip = request.client.host if request.client else "Unknown"

    mongodb = get_mongodb_service()
    collection = mongodb.db[COLLECTION_DOCUMENTS]

    # Build match stage based on filters
    match_stage = {"chunk_index": 0}  # Only count first chunks (unique docs)

    if department:
        match_stage["department"] = department
    if doc_type:
        match_stage["type"] = doc_type
    if subject:
        match_stage["subject"] = subject

    # Pipeline for count
    count_pipeline = [
        {"$match": match_stage},
        {"$count": "total_documents"}
    ]

    # Pipeline for document titles
    titles_pipeline = [
        {"$match": match_stage},
        {"$group": {"_id": "$title"}},
        {"$project": {"title": "$_id", "_id": 0}},
        {"$sort": {"title": 1}}
    ]

    # Execute pipelines
    count_result = await collection.aggregate(count_pipeline).to_list(length=1)
    titles_result = await collection.aggregate(titles_pipeline).to_list(length=1000)

    total_count = count_result[0]["total_documents"] if count_result else 0
    titles = [doc["title"] for doc in titles_result if doc.get("title")]

    log_pipeline("DOCUMENT", user_ip, "Document aggregation",
                 details={
                     "department": department,
                     "type": doc_type,
                     "subject": subject,
                     "count": total_count
                 })

    return {
        "success": True,
        "total_documents": total_count,
        "titles": titles,
        "filters": {
            "department": department,
            "type": doc_type,
            "subject": subject
        }
    }


@router.get("/by-title")
async def get_document_by_title(
    request: Request,
    title: str = Query(..., description="Document title to fetch")
):
    """Get all chunks of a document by title and return combined content"""
    user_ip = request.client.host if request.client else "Unknown"

    mongodb = get_mongodb_service()
    collection = mongodb.db[COLLECTION_DOCUMENTS]

    # Find all chunks for this document, sorted by chunk_index
    cursor = collection.find({"title": title}).sort("chunk_index", 1)
    chunks = await cursor.to_list(length=1000)

    log_pipeline("DOCUMENT", user_ip, "Document retrieved by title",
                 title,
                 details={"found": len(chunks) > 0, "chunks": len(chunks)})

    if not chunks:
        raise HTTPException(status_code=404, detail="Document not found")

    # Combine all chunk content
    combined_content = ""
    for chunk in chunks:
        if chunk.get("content"):
            combined_content += chunk["content"] + "\n\n"

    # Return metadata from first chunk plus combined content
    first_chunk = chunks[0]
    return {
        "title": title,
        "department": first_chunk.get("department"),
        "type": first_chunk.get("type"),
        "subject": first_chunk.get("subject"),
        "total_chunks": len(chunks),
        "content": combined_content.strip(),
        "created_at": first_chunk.get("created_at"),
        "file_size": first_chunk.get("file_size")
    }


@router.delete("/by-title")
async def delete_document_by_title(
    request: Request,
    title: str = Query(..., description="Document title to delete")
):
    """Delete all chunks of a document by title"""
    user_ip = request.client.host if request.client else "Unknown"

    mongodb = get_mongodb_service()
    collection = mongodb.db[COLLECTION_DOCUMENTS]

    # Delete all chunks with this title
    result = await collection.delete_many({"title": title})

    log_pipeline("DOCUMENT", user_ip, "Document deleted by title",
                 title,
                 details={"deleted_count": result.deleted_count})

    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Document not found")

    return {
        "success": True,
        "title": title,
        "deleted_chunks": result.deleted_count,
        "message": f"Deleted {result.deleted_count} chunks for '{title}'"
    }


@router.get("/stats/summary", response_model=StatsResponse)
async def get_document_stats(request: Request):
    """Get document collection statistics"""
    user_ip = request.client.host if request.client else "Unknown"

    log_pipeline("DOCUMENT", user_ip, "Document stats requested")

    mongodb = get_mongodb_service()
    stats = await mongodb.get_document_stats()
    return StatsResponse(**stats)


@router.get("/{document_id}")
async def get_document(document_id: str, request: Request):
    """Get a document by ID"""
    user_ip = request.client.host if request.client else "Unknown"

    mongodb = get_mongodb_service()
    doc = await mongodb.get_document(document_id)

    log_pipeline("DOCUMENT", user_ip, "Document retrieved",
                 document_id,
                 details={"found": doc is not None})

    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.get("")
async def list_documents(
    request: Request,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """List all documents"""
    user_ip = request.client.host if request.client else "Unknown"

    mongodb = get_mongodb_service()
    result = await mongodb.list_documents(limit=limit, offset=offset)

    log_pipeline("DOCUMENT", user_ip, "Documents listed",
                 details={
                     "limit": limit,
                     "offset": offset,
                     "count": len(result)
                 })

    return result


@router.patch("/{document_id}")
async def update_document_metadata(document_id: str, update: DocumentUpdate, request: Request):
    """Update document metadata without re-embedding"""
    user_ip = request.client.host if request.client else "Unknown"

    mongodb = get_mongodb_service()
    result = await mongodb.update_document_metadata(
        document_id=document_id,
        title=update.title,
        department=update.department,
        doc_type=update.type,
        subject=update.subject,
        tags=update.tags,
        metadata=update.metadata
    )

    log_pipeline("DOCUMENT", user_ip, "Document metadata updated",
                 document_id,
                 details={"success": result.get("success", False)})

    if not result.get("success"):
        raise HTTPException(status_code=404, detail="Document not found")
    return result


@router.delete("/{document_id}", response_model=DeleteResponse)
async def delete_document(document_id: str, request: Request):
    """Delete a document and all its chunks"""
    user_ip = request.client.host if request.client else "Unknown"

    mongodb = get_mongodb_service()
    result = await mongodb.delete_document(document_id)

    log_pipeline("DOCUMENT", user_ip, "Document deleted",
                 document_id,
                 details={"success": result.get("success", False)})

    return DeleteResponse(**result)
