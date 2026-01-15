import EWRAIDatabase from '../src/db/EWRAIDatabase.js';

async function initDatabase() {
    const db = new EWRAIDatabase();
    await db.initialize();

    const settings = db.getAllSettings();

    console.log('Database initialized with settings:');
    console.log('  MongoDBUri:', settings.MongoDBUri || 'mongodb://localhost:27017');
    console.log('  MongoDBDatabase:', settings.MongoDBDatabase || 'rag_server');
    console.log('  PythonServiceUrl:', settings.PythonServiceUrl || 'http://localhost:8001');
    console.log('  LlamaCppHost:', settings.LlamaCppHost || 'http://localhost:11434');
    console.log('  NodeServerPort:', settings.NodeServerPort || '3000');

    db.close();
}

initDatabase().catch(err => {
    console.error('Failed to initialize database:', err.message);
    process.exit(1);
});
