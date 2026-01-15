"""
Category routes for document classification management.

Provides endpoints for:
- Category CRUD operations (departments, types, subjects)
"""
import logging
import sys
from pathlib import Path

from fastapi import APIRouter, Request, HTTPException

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_models import (
    CategoryBase,
    CategoryCreate,
    CategoryUpdate,
    CategoryResponse,
    CategoryListResponse
)
from category_service import get_category_service
from log_service import log_pipeline, log_error

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/categories", tags=["Categories"])

@router.get("")
async def get_all_categories(request: Request):
    """
    Get all categories of all types.

    Returns:
        Object with departments, types, and subjects arrays
    """
    user_ip = request.client.host if request.client else "Unknown"

    try:
        category_service = get_category_service()

        # Fetch all three category types
        departments = await category_service.get_all("departments")
        types = await category_service.get_all("types")
        subjects = await category_service.get_all("subjects")

        # Map display_order to order for response
        def map_categories(cats):
            result = []
            for cat in cats:
                if "display_order" in cat:
                    cat["order"] = cat.pop("display_order")
                result.append(cat)
            return result

        log_pipeline("CATEGORY", user_ip, "All categories retrieved",
                     details={"departments": len(departments), "types": len(types), "subjects": len(subjects)})

        return {
            "success": True,
            "departments": map_categories(departments),
            "types": map_categories(types),
            "subjects": map_categories(subjects)
        }
    except Exception as e:
        log_error("CATEGORY", user_ip, "Failed to get all categories", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{category_type}", response_model=CategoryListResponse)
async def get_categories(category_type: str, request: Request):
    """
    Get all categories of a specific type.

    Args:
        category_type: One of "departments", "types", or "subjects"

    Returns:
        List of categories with id, name, description, and order
    """
    user_ip = request.client.host if request.client else "Unknown"

    # Validate category_type
    valid_types = ["departments", "types", "subjects"]
    if category_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid category_type: {category_type}. Must be one of: {valid_types}"
        )

    try:
        category_service = get_category_service()
        categories = await category_service.get_all(category_type)

        log_pipeline("CATEGORY", user_ip, f"Categories retrieved",
                     details={"category_type": category_type, "count": len(categories)})

        # Map display_order to order for response
        mapped_categories = []
        for cat in categories:
            if "display_order" in cat:
                cat["order"] = cat.pop("display_order")
            mapped_categories.append(CategoryBase(**cat))

        return CategoryListResponse(
            success=True,
            categories=mapped_categories
        )
    except Exception as e:
        log_error("CATEGORY", user_ip, f"Failed to get categories", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{category_type}", response_model=CategoryResponse)
async def create_category(category_type: str, category: CategoryCreate, request: Request):
    """
    Create a new category.

    Args:
        category_type: One of "departments", "types", or "subjects"
        category: Category data with id, name, description, order

    Returns:
        The created category

    Raises:
        409: If a category with the same ID already exists
    """
    user_ip = request.client.host if request.client else "Unknown"

    # Validate category_type
    valid_types = ["departments", "types", "subjects"]
    if category_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid category_type: {category_type}. Must be one of: {valid_types}"
        )

    try:
        category_service = get_category_service()
        result = await category_service.create(
            table_name=category_type,
            category_id=category.id,
            name=category.name,
            description=category.description,
            display_order=category.order
        )

        log_pipeline("CATEGORY", user_ip, f"Category created",
                     details={"category_type": category_type, "id": category.id})

        # Map display_order to order for response
        created = result.get("category", {})
        if "display_order" in created:
            created["order"] = created.pop("display_order")

        return CategoryResponse(
            success=True,
            category=CategoryBase(**created),
            message=f"Category '{category.id}' created successfully"
        )
    except ValueError as e:
        # Category already exists
        log_error("CATEGORY", user_ip, f"Category creation failed - duplicate ID", str(e))
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        log_error("CATEGORY", user_ip, f"Failed to create category", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{category_type}/{category_id}", response_model=CategoryResponse)
async def update_category(
    category_type: str,
    category_id: str,
    update: CategoryUpdate,
    request: Request
):
    """
    Update an existing category.

    Args:
        category_type: One of "departments", "types", or "subjects"
        category_id: ID of the category to update
        update: Fields to update (name, description, order)

    Returns:
        The updated category

    Raises:
        404: If category not found
    """
    user_ip = request.client.host if request.client else "Unknown"

    # Validate category_type
    valid_types = ["departments", "types", "subjects"]
    if category_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid category_type: {category_type}. Must be one of: {valid_types}"
        )

    try:
        category_service = get_category_service()
        result = await category_service.update(
            table_name=category_type,
            category_id=category_id,
            name=update.name,
            description=update.description,
            display_order=update.order
        )

        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"Category '{category_id}' not found in {category_type}"
            )

        log_pipeline("CATEGORY", user_ip, f"Category updated",
                     details={"category_type": category_type, "id": category_id})

        # Map display_order to order for response
        updated = result.get("category", {})
        if "display_order" in updated:
            updated["order"] = updated.pop("display_order")

        return CategoryResponse(
            success=True,
            category=CategoryBase(**updated),
            message=f"Category '{category_id}' updated successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        log_error("CATEGORY", user_ip, f"Failed to update category", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{category_type}/{category_id}", response_model=CategoryResponse)
async def delete_category(category_type: str, category_id: str, request: Request):
    """
    Delete a category.

    Args:
        category_type: One of "departments", "types", or "subjects"
        category_id: ID of the category to delete

    Returns:
        Success message

    Raises:
        404: If category not found
    """
    user_ip = request.client.host if request.client else "Unknown"

    # Validate category_type
    valid_types = ["departments", "types", "subjects"]
    if category_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid category_type: {category_type}. Must be one of: {valid_types}"
        )

    try:
        category_service = get_category_service()
        result = await category_service.delete(category_type, category_id)

        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"Category '{category_id}' not found in {category_type}"
            )

        log_pipeline("CATEGORY", user_ip, f"Category deleted",
                     details={"category_type": category_type, "id": category_id})

        return CategoryResponse(
            success=True,
            message=f"Category '{category_id}' deleted successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        log_error("CATEGORY", user_ip, f"Failed to delete category", str(e))
        raise HTTPException(status_code=500, detail=str(e))
