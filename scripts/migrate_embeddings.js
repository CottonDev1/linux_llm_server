// Embedding Migration Script for mongosh
// Run this in MongoDB Compass shell (Open MongoDB Shell button)
//
// Usage: Copy and paste this entire script into the mongosh shell
// Or save to file and run: load('/path/to/migrate_embeddings.js')

const EMBEDDING_URL = "http://EWRSPT-AI:8083/embedding";

// Collections and their text fields
const COLLECTIONS = {
    "documents": "content",
    "code_classes": "embedding_text",
    "code_context": "embedding_text",
    "code_methods": "embedding_text",
    "sql_stored_procedures": "embedding_text",
    "sql_examples": "embedding_text",
    "sql_schema_context": "embedding_text"
};

// Function to get embedding from llama.cpp server
async function getEmbedding(text) {
    const response = await fetch(EMBEDDING_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: text.substring(0, 8000) })
    });

    if (!response.ok) {
        throw new Error(`Embedding failed: ${response.statusText}`);
    }

    const data = await response.json();
    // llama.cpp format: [{"index": 0, "embedding": [[...floats...]]}]
    return data[0].embedding[0];
}

// Migrate a single collection
async function migrateCollection(collName, textField) {
    const coll = db.getCollection(collName);
    const total = coll.countDocuments({ vector: { $exists: true } });

    if (total === 0) {
        print(`  ${collName}: No documents with vectors, skipping`);
        return 0;
    }

    print(`  ${collName}: Migrating ${total} documents...`);

    let updated = 0;
    let errors = 0;

    const cursor = coll.find({ vector: { $exists: true } });

    while (cursor.hasNext()) {
        const doc = cursor.next();
        let text = doc[textField];

        // Fallback to common fields
        if (!text) {
            text = doc.embedding_text || doc.content || doc.text || doc.description;
        }

        if (!text) {
            print(`    Warning: No text found for doc ${doc._id}`);
            errors++;
            continue;
        }

        try {
            const embedding = await getEmbedding(text);
            coll.updateOne(
                { _id: doc._id },
                { $set: { vector: embedding } }
            );
            updated++;

            if (updated % 100 === 0) {
                print(`    Progress: ${updated}/${total} (${Math.floor(100*updated/total)}%)`);
            }
        } catch (e) {
            print(`    Error on ${doc._id}: ${e.message}`);
            errors++;
        }
    }

    print(`  ${collName}: Done - ${updated} updated, ${errors} errors`);
    return updated;
}

// Main migration function
async function migrate() {
    print("============================================================");
    print("Embedding Migration: 384 -> 768 dimensions");
    print("============================================================");
    print(`Embedding Service: ${EMBEDDING_URL}`);
    print("");

    // Test embedding service
    print("Testing embedding service...");
    try {
        const testEmbed = await getEmbedding("test");
        print(`  OK - Embedding dimensions: ${testEmbed.length}`);
        if (testEmbed.length !== 768) {
            print(`  WARNING: Expected 768 dimensions, got ${testEmbed.length}`);
        }
    } catch (e) {
        print(`  FAILED: ${e.message}`);
        print("  Make sure the embedding service is running on EWRSPT-AI:8083");
        return;
    }

    print("");
    print("Starting migration...");
    print("");

    use("rag_server");

    let totalUpdated = 0;
    for (const [collName, textField] of Object.entries(COLLECTIONS)) {
        try {
            const updated = await migrateCollection(collName, textField);
            totalUpdated += updated;
        } catch (e) {
            print(`  ${collName}: FAILED - ${e.message}`);
        }
    }

    print("");
    print("============================================================");
    print(`Migration complete. Total documents updated: ${totalUpdated}`);
    print("============================================================");
}

// Run the migration
migrate();
