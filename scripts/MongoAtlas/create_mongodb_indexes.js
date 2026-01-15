/**
 * MongoDB Index Creation Script
 * Document Retrieval Pipeline Migration
 *
 * Usage:
 *   mongosh mongodb://EWRSPT-AI:27018/rag_server < create_mongodb_indexes.js
 *
 * Or connect and run:
 *   mongosh mongodb://EWRSPT-AI:27018/rag_server
 *   load('create_mongodb_indexes.js')
 */

print("========================================");
print("MongoDB Index Creation for Document Retrieval Pipeline");
print("========================================\n");

// Use the rag_server database
db = db.getSiblingDB('rag_server');

print("Database: " + db.getName());
print("Collections: " + db.getCollectionNames().join(', '));
print("");

// ============================================================================
// DOCUMENTS COLLECTION INDEXES
// ============================================================================

print("Creating indexes for 'documents' collection...\n");

// 1. Text Index for BM25 (Hybrid Search)
print("1. Creating text index for BM25 search...");
try {
    db.documents.createIndex(
        {
            "title": "text",
            "content": "text",
            "section_title": "text"
        },
        {
            weights: {
                title: 10,
                section_title: 5,
                content: 1
            },
            name: "documents_text_index",
            default_language: "english"
        }
    );
    print("   ✓ Created: documents_text_index");
} catch (e) {
    if (e.code === 85) {
        print("   ℹ️  Text index already exists, skipping");
    } else {
        print("   ❌ Error: " + e.message);
    }
}

// 2. Compound Index for Filtering
print("\n2. Creating compound filter index...");
try {
    db.documents.createIndex(
        {
            "department": 1,
            "type": 1,
            "hierarchy_level": 1,
            "upload_date": -1
        },
        {
            name: "documents_filter_index"
        }
    );
    print("   ✓ Created: documents_filter_index");
} catch (e) {
    if (e.code === 85 || e.codeName === "IndexOptionsConflict") {
        print("   ℹ️  Filter index already exists, skipping");
    } else {
        print("   ❌ Error: " + e.message);
    }
}

// 3. Parent-Child Relationship Index
print("\n3. Creating parent-child relationship index...");
try {
    db.documents.createIndex(
        {
            "parent_id": 1,
            "chunk_index": 1
        },
        {
            name: "documents_parent_child_index"
        }
    );
    print("   ✓ Created: documents_parent_child_index");
} catch (e) {
    if (e.code === 85 || e.codeName === "IndexOptionsConflict") {
        print("   ℹ️  Parent-child index already exists, skipping");
    } else {
        print("   ❌ Error: " + e.message);
    }
}

// 4. Tags Index
print("\n4. Creating tags index...");
try {
    db.documents.createIndex(
        {
            "tags": 1
        },
        {
            name: "documents_tags_index"
        }
    );
    print("   ✓ Created: documents_tags_index");
} catch (e) {
    if (e.code === 85 || e.codeName === "IndexOptionsConflict") {
        print("   ℹ️  Tags index already exists, skipping");
    } else {
        print("   ❌ Error: " + e.message);
    }
}

// 5. Entities Index (for new schema)
print("\n5. Creating entities index...");
try {
    db.documents.createIndex(
        {
            "entities": 1
        },
        {
            name: "documents_entities_index"
        }
    );
    print("   ✓ Created: documents_entities_index");
} catch (e) {
    if (e.code === 85 || e.codeName === "IndexOptionsConflict") {
        print("   ℹ️  Entities index already exists, skipping");
    } else {
        print("   ❌ Error: " + e.message);
    }
}

// 6. Quality Metrics Index
print("\n6. Creating quality metrics index...");
try {
    db.documents.createIndex(
        {
            "relevance_feedback_score": -1,
            "access_count": -1
        },
        {
            name: "documents_quality_index"
        }
    );
    print("   ✓ Created: documents_quality_index");
} catch (e) {
    if (e.code === 85 || e.codeName === "IndexOptionsConflict") {
        print("   ℹ️  Quality index already exists, skipping");
    } else {
        print("   ❌ Error: " + e.message);
    }
}

// 7. Source File Hash Index (for deduplication)
print("\n7. Creating source file hash index...");
try {
    db.documents.createIndex(
        {
            "source_file_hash": 1
        },
        {
            name: "documents_source_hash_index"
        }
    );
    print("   ✓ Created: documents_source_hash_index");
} catch (e) {
    if (e.code === 85 || e.codeName === "IndexOptionsConflict") {
        print("   ℹ️  Source hash index already exists, skipping");
    } else {
        print("   ❌ Error: " + e.message);
    }
}

// ============================================================================
// FEEDBACK COLLECTION INDEXES
// ============================================================================

print("\n\nCreating indexes for 'feedback' collection...\n");

// 1. Query Embedding Index (for semantic similarity)
print("1. Creating query embedding index...");
try {
    db.feedback.createIndex(
        {
            "query_hash": 1,
            "created_at": -1
        },
        {
            name: "feedback_query_hash_index"
        }
    );
    print("   ✓ Created: feedback_query_hash_index");
} catch (e) {
    if (e.code === 85 || e.codeName === "IndexOptionsConflict") {
        print("   ℹ️  Query hash index already exists, skipping");
    } else {
        print("   ❌ Error: " + e.message);
    }
}

// 2. Feedback Type Index
print("\n2. Creating feedback type index...");
try {
    db.feedback.createIndex(
        {
            "feedback_type": 1,
            "created_at": -1
        },
        {
            name: "feedback_type_index"
        }
    );
    print("   ✓ Created: feedback_type_index");
} catch (e) {
    if (e.code === 85 || e.codeName === "IndexOptionsConflict") {
        print("   ℹ️  Feedback type index already exists, skipping");
    } else {
        print("   ❌ Error: " + e.message);
    }
}

// 3. Document Reference Index
print("\n3. Creating document reference index...");
try {
    db.feedback.createIndex(
        {
            "retrieved_documents": 1
        },
        {
            name: "feedback_documents_index"
        }
    );
    print("   ✓ Created: feedback_documents_index");
} catch (e) {
    if (e.code === 85 || e.codeName === "IndexOptionsConflict") {
        print("   ℹ️  Document reference index already exists, skipping");
    } else {
        print("   ❌ Error: " + e.message);
    }
}

// ============================================================================
// SQL COLLECTIONS INDEXES (Existing, verify they exist)
// ============================================================================

print("\n\nVerifying indexes for SQL-related collections...\n");

// SQL Examples
print("1. Checking sql_examples indexes...");
try {
    db.sql_examples.createIndex(
        {
            "database_name": 1,
            "created_at": -1
        },
        {
            name: "sql_examples_db_index"
        }
    );
    print("   ✓ Created/verified: sql_examples_db_index");
} catch (e) {
    if (e.code === 85 || e.codeName === "IndexOptionsConflict") {
        print("   ℹ️  Already exists");
    } else {
        print("   ❌ Error: " + e.message);
    }
}

// SQL Failed Queries
print("\n2. Checking sql_failed_queries indexes...");
try {
    db.sql_failed_queries.createIndex(
        {
            "database_name": 1,
            "created_at": -1
        },
        {
            name: "sql_failed_db_index"
        }
    );
    print("   ✓ Created/verified: sql_failed_db_index");
} catch (e) {
    if (e.code === 85 || e.codeName === "IndexOptionsConflict") {
        print("   ℹ️  Already exists");
    } else {
        print("   ❌ Error: " + e.message);
    }
}

// SQL Corrections
print("\n3. Checking sql_corrections indexes...");
try {
    db.sql_corrections.createIndex(
        {
            "database_name": 1,
            "verified": 1,
            "created_at": -1
        },
        {
            name: "sql_corrections_index"
        }
    );
    print("   ✓ Created/verified: sql_corrections_index");
} catch (e) {
    if (e.code === 85 || e.codeName === "IndexOptionsConflict") {
        print("   ℹ️  Already exists");
    } else {
        print("   ❌ Error: " + e.message);
    }
}

// ============================================================================
// SUMMARY
// ============================================================================

print("\n========================================");
print("Index Creation Summary");
print("========================================\n");

// Get all indexes for documents collection
print("Documents Collection Indexes:");
db.documents.getIndexes().forEach(function(index) {
    print("  - " + index.name + " (" + JSON.stringify(index.key) + ")");
});

print("\nFeedback Collection Indexes:");
if (db.feedback.exists()) {
    db.feedback.getIndexes().forEach(function(index) {
        print("  - " + index.name + " (" + JSON.stringify(index.key) + ")");
    });
} else {
    print("  ℹ️  Collection does not exist yet");
}

print("\n========================================");
print("Index creation complete!");
print("========================================\n");

// Check MongoDB version and vector search capability
print("MongoDB Version: " + db.version());

// Note about vector search index
print("\n📝 NOTE: Vector search index must be created via MongoDB Atlas UI or MongoDB 8.2+ shell");
print("   For Atlas: Database > Search > Create Vector Search Index");
print("   Index configuration:");
print("   {");
print('     "type": "vectorSearch",');
print('     "fields": [{');
print('       "type": "vector",');
print('       "path": "vector",');
print('       "numDimensions": 384,  // or 1024 for upgraded model');
print('       "similarity": "cosine",');
print('       "quantization": "scalar"  // Optional: 4x storage reduction');
print("     }]");
print("   }");

print("\n✓ Index creation script completed successfully");
