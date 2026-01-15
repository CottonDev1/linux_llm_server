/**
 * Python API Client
 *
 * Thin HTTP client that forwards all data operations to the Python service.
 * Node.js server should only handle HTTP routing - all business logic is in Python.
 */

const PYTHON_SERVICE_URL = process.env.PYTHON_SERVICE_URL || 'http://localhost:8001';

/**
 * SQLKnowledgeDB - HTTP client wrapper for Python SQL RAG endpoints
 */
export class SQLKnowledgeDB {
  constructor(options = {}) {
    this.pythonServiceUrl = options.pythonServiceUrl || PYTHON_SERVICE_URL;
    this.isInitialized = false;
  }

  async initialize() {
    if (this.isInitialized) return;

    try {
      const response = await fetch(`${this.pythonServiceUrl}/status`);
      if (!response.ok) {
        throw new Error(`Python service returned ${response.status}`);
      }
      this.isInitialized = true;
      console.log(`✅ SQLKnowledgeDB connected to Python service at ${this.pythonServiceUrl}`);
    } catch (error) {
      throw new Error(`Failed to connect to Python service: ${error.message}`);
    }
  }

  // ========================================================================
  // SQL Examples (Few-Shot Learning)
  // ========================================================================

  async storeExample(database, prompt, sql, response = null) {
    const res = await fetch(`${this.pythonServiceUrl}/sql/examples`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ database, prompt, sql, response })
    });

    if (!res.ok) {
      throw new Error(`Failed to store example: ${await res.text()}`);
    }
    return (await res.json()).example_id;
  }

  async searchExamples(query, options = {}) {
    const params = new URLSearchParams({
      query,
      ...(options.database && { database: options.database }),
      ...(options.limit && { limit: options.limit.toString() })
    });

    const res = await fetch(`${this.pythonServiceUrl}/sql/examples/search?${params}`);
    if (!res.ok) {
      throw new Error(`Failed to search examples: ${await res.text()}`);
    }
    return await res.json();
  }

  // ========================================================================
  // SQL Failed Queries (Error Learning)
  // ========================================================================

  async storeFailedQuery(database, prompt, sql, error, context = {}) {
    const res = await fetch(`${this.pythonServiceUrl}/sql/failed-queries`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        database,
        prompt,
        sql,
        error,
        tables_involved: context.tables || [],
        // Pass correction references if this failed query was corrected
        correction_id: context.correctionId || null,
        corrected_sql: context.correctedSql || null
      })
    });

    if (!res.ok) {
      throw new Error(`Failed to store failed query: ${await res.text()}`);
    }
    return (await res.json()).failed_id;
  }

  async searchFailedQueries(query, options = {}) {
    const params = new URLSearchParams({
      query,
      ...(options.database && { database: options.database }),
      ...(options.limit && { limit: options.limit.toString() })
    });

    const res = await fetch(`${this.pythonServiceUrl}/sql/failed-queries/search?${params}`);
    if (!res.ok) {
      throw new Error(`Failed to search failed queries: ${await res.text()}`);
    }
    return await res.json();
  }

  // ========================================================================
  // SQL Schema Context
  // ========================================================================

  async storeSchemaContext(database, tableName, schemaInfo) {
    const res = await fetch(`${this.pythonServiceUrl}/sql/schema-context`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        database,
        table_name: tableName,
        schema_info: schemaInfo
      })
    });

    if (!res.ok) {
      throw new Error(`Failed to store schema context: ${await res.text()}`);
    }
    return (await res.json()).schema_id;
  }

  async searchSchemaContext(query, options = {}) {
    const params = new URLSearchParams({
      query,
      ...(options.database && { database: options.database }),
      ...(options.limit && { limit: options.limit.toString() })
    });

    const url = `${this.pythonServiceUrl}/sql/schema-context/search?${params}`;
    console.log(`   [DEBUG] searchSchemaContext URL: ${url}`);

    const res = await fetch(url);
    console.log(`   [DEBUG] Response status: ${res.status}`);

    if (!res.ok) {
      const errText = await res.text();
      console.log(`   [DEBUG] Error response: ${errText}`);
      throw new Error(`Failed to search schema context: ${errText}`);
    }
    const results = await res.json();
    console.log(`   [DEBUG] Results count: ${Array.isArray(results) ? results.length : 'not array'}`);
    return results;
  }

  async getSchemaByTable(database, tableName) {
    const res = await fetch(`${this.pythonServiceUrl}/sql/schema-context/${encodeURIComponent(database)}/${encodeURIComponent(tableName)}`);
    if (!res.ok) {
      if (res.status === 404) {
        return null; // Table not found
      }
      throw new Error(`Failed to get schema by table: ${await res.text()}`);
    }
    return await res.json();
  }

  // ========================================================================
  // SQL Stored Procedures
  // ========================================================================

  async storeStoredProcedure(database, procedureName, procedureInfo) {
    const res = await fetch(`${this.pythonServiceUrl}/sql/stored-procedures`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        database,
        procedure_name: procedureName,
        procedure_info: procedureInfo
      })
    });

    if (!res.ok) {
      throw new Error(`Failed to store stored procedure: ${await res.text()}`);
    }
    return (await res.json()).procedure_id;
  }

  async searchStoredProcedures(query, options = {}) {
    const params = new URLSearchParams({
      query,
      ...(options.database && { database: options.database }),
      ...(options.limit && { limit: options.limit.toString() })
    });

    const res = await fetch(`${this.pythonServiceUrl}/sql/stored-procedures/search?${params}`);
    if (!res.ok) {
      throw new Error(`Failed to search stored procedures: ${await res.text()}`);
    }
    return await res.json();
  }

  // ========================================================================
  // Comprehensive Context (Main RAG Entry Point)
  // ========================================================================

  async getComprehensiveContext(query, database, options = {}) {
    const params = new URLSearchParams({
      query,
      database,
      ...(options.schemaLimit && { schema_limit: options.schemaLimit.toString() }),
      ...(options.exampleLimit && { example_limit: options.exampleLimit.toString() }),
      ...(options.failedLimit && { failed_limit: options.failedLimit.toString() }),
      ...(options.spLimit && { sp_limit: options.spLimit.toString() })
    });

    const res = await fetch(`${this.pythonServiceUrl}/sql/comprehensive-context?${params}`);
    if (!res.ok) {
      throw new Error(`Failed to get comprehensive context: ${await res.text()}`);
    }
    return await res.json();
  }

  // ========================================================================
  // Store Successful Example (alias for storeExample with response string)
  // ========================================================================

  async storeSuccessfulExample(database, prompt, sql, response = 'Auto-saved successful query') {
    return await this.storeExample(database, prompt, sql, response);
  }

  // ========================================================================
  // Schema Existence Check
  // ========================================================================

  async getDatabaseStats(database) {
    try {
      const res = await fetch(`${this.pythonServiceUrl}/sql/database-stats/${encodeURIComponent(database)}`);
      if (!res.ok) {
        console.error(`   [DEBUG] getDatabaseStats failed: ${res.status}`);
        return { schema_count: 0, procedure_count: 0, has_data: false };
      }
      return await res.json();
    } catch (error) {
      console.error(`   [DEBUG] getDatabaseStats error: ${error.message}`);
      return { schema_count: 0, procedure_count: 0, has_data: false };
    }
  }

  async checkSchemaExists(database) {
    try {
      const stats = await this.getDatabaseStats(database);
      return stats.has_data;
    } catch (error) {
      return false;
    }
  }

  async getSchemaCount(database) {
    try {
      const stats = await this.getDatabaseStats(database);
      return stats.schema_count || 0;
    } catch (error) {
      return 0;
    }
  }

  async getStoredProcedureCount(database) {
    try {
      const stats = await this.getDatabaseStats(database);
      return stats.procedure_count || 0;
    } catch (error) {
      return 0;
    }
  }

  // ========================================================================
  // Statistics
  // ========================================================================

  async getStatistics() {
    const res = await fetch(`${this.pythonServiceUrl}/sql/rag-stats`);
    if (!res.ok) {
      throw new Error(`Failed to get statistics: ${await res.text()}`);
    }
    return await res.json();
  }

  // ========================================================================
  // Analyzed Databases List
  // ========================================================================

  /**
   * Get list of databases that have been analyzed and stored in MongoDB
   * @returns {Promise<string[]>} List of database names
   */
  async getAnalyzedDatabases() {
    try {
      const res = await fetch(`${this.pythonServiceUrl}/sql/pipeline-stats`);
      if (!res.ok) {
        throw new Error(`Failed to get pipeline stats: ${await res.text()}`);
      }
      const stats = await res.json();

      // Get unique databases from schema and procedures stats
      const databases = new Set();

      if (stats.schema_by_database) {
        Object.keys(stats.schema_by_database).forEach(db => databases.add(db));
      }

      if (stats.procedures_by_database) {
        Object.keys(stats.procedures_by_database).forEach(db => databases.add(db));
      }

      if (stats.examples_by_database) {
        Object.keys(stats.examples_by_database).forEach(db => databases.add(db));
      }

      return Array.from(databases).sort();
    } catch (error) {
      console.error('Failed to get analyzed databases:', error.message);
      return [];
    }
  }
}

/**
 * DocumentationDB - HTTP client wrapper for Python document endpoints
 */
export class DocumentationDB {
  constructor(options = {}) {
    this.pythonServiceUrl = options.pythonServiceUrl || PYTHON_SERVICE_URL;
    this.isInitialized = false;
  }

  async initialize() {
    if (this.isInitialized) return;

    try {
      const response = await fetch(`${this.pythonServiceUrl}/status`);
      if (!response.ok) {
        throw new Error(`Python service returned ${response.status}`);
      }
      this.isInitialized = true;
    } catch (error) {
      throw new Error(`Failed to connect to Python service: ${error.message}`);
    }
  }

  async store(title, content, metadata = {}) {
    const res = await fetch(`${this.pythonServiceUrl}/documents`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        title,
        content,
        department: metadata.department || 'general',
        type: metadata.type || 'documentation',
        subject: metadata.subject,
        file_name: metadata.fileName,
        file_size: metadata.fileSize || 0,
        tags: metadata.tags || [],
        metadata: metadata
      })
    });

    if (!res.ok) {
      throw new Error(`Failed to store document: ${await res.text()}`);
    }
    return await res.json();
  }

  async search(query, options = {}) {
    const params = new URLSearchParams({
      query,
      ...(options.limit && { limit: options.limit.toString() }),
      ...(options.department && { department: options.department }),
      ...(options.type && { type: options.type })
    });

    const res = await fetch(`${this.pythonServiceUrl}/documents/search?${params}`);
    if (!res.ok) {
      throw new Error(`Failed to search documents: ${await res.text()}`);
    }
    return await res.json();
  }

  async createEmbedding(text) {
    // For backward compatibility - search with the text to get similar docs
    return await this.search(text, { limit: 5 });
  }

  async getStats() {
    const res = await fetch(`${this.pythonServiceUrl}/documents/stats/summary`);
    if (!res.ok) {
      throw new Error(`Failed to get stats: ${await res.text()}`);
    }
    return await res.json();
  }

  async listDocuments(limit = 100, offset = 0) {
    const params = new URLSearchParams({
      limit: limit.toString(),
      offset: offset.toString()
    });
    const res = await fetch(`${this.pythonServiceUrl}/documents?${params}`);
    if (!res.ok) {
      throw new Error(`Failed to list documents: ${await res.text()}`);
    }
    return await res.json();
  }

  async getDocument(documentId) {
    const res = await fetch(`${this.pythonServiceUrl}/documents/${encodeURIComponent(documentId)}`);
    if (!res.ok) {
      if (res.status === 404) {
        return null;
      }
      throw new Error(`Failed to get document: ${await res.text()}`);
    }
    return await res.json();
  }

  async deleteDocument(documentId) {
    const res = await fetch(`${this.pythonServiceUrl}/documents/${encodeURIComponent(documentId)}`, {
      method: 'DELETE'
    });
    if (!res.ok) {
      throw new Error(`Failed to delete document: ${await res.text()}`);
    }
    return await res.json();
  }

  async updateDocument(documentId, updates = {}) {
    const res = await fetch(`${this.pythonServiceUrl}/documents/${encodeURIComponent(documentId)}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        title: updates.title,
        department: updates.department,
        type: updates.type,
        subject: updates.subject,
        tags: updates.tags,
        metadata: updates.metadata
      })
    });
    if (!res.ok) {
      throw new Error(`Failed to update document: ${await res.text()}`);
    }
    return await res.json();
  }

  async addDocuments(documents) {
    // Add multiple documents - Python service handles chunking and embedding
    const results = [];
    for (const doc of documents) {
      const result = await this.store(
        doc.title,
        doc.content,
        {
          department: doc.department || doc.category || 'general',
          type: doc.type || doc.documentType || 'documentation',
          subject: doc.subject,
          fileName: doc.fileName,
          fileSize: doc.fileSize || 0,
          tags: doc.tags || [],
          ...doc.metadata
        }
      );
      results.push(result);
    }
    return results;
  }
}

/**
 * CodeContextDB - HTTP client wrapper for Python code context endpoints
 */
export class CodeContextDB {
  constructor(options = {}) {
    this.pythonServiceUrl = options.pythonServiceUrl || PYTHON_SERVICE_URL;
    this.isInitialized = false;
  }

  async initialize() {
    if (this.isInitialized) return;

    try {
      const response = await fetch(`${this.pythonServiceUrl}/status`);
      if (!response.ok) {
        throw new Error(`Python service returned ${response.status}`);
      }
      this.isInitialized = true;
    } catch (error) {
      throw new Error(`Failed to connect to Python service: ${error.message}`);
    }
  }

  async store(documentId, content, metadata = {}) {
    const res = await fetch(`${this.pythonServiceUrl}/code-context`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        document_id: documentId,
        content,
        metadata
      })
    });

    if (!res.ok) {
      throw new Error(`Failed to store code context: ${await res.text()}`);
    }
    return (await res.json()).document_id;
  }

  async search(query, options = {}) {
    const params = new URLSearchParams({
      query,
      ...(options.limit && { limit: options.limit.toString() }),
      ...(options.project && { project: options.project })
    });

    const res = await fetch(`${this.pythonServiceUrl}/code-context/search?${params}`);
    if (!res.ok) {
      throw new Error(`Failed to search code context: ${await res.text()}`);
    }
    return await res.json();
  }

  async getStats() {
    const res = await fetch(`${this.pythonServiceUrl}/code-context/stats/summary`);
    if (!res.ok) {
      throw new Error(`Failed to get stats: ${await res.text()}`);
    }
    return await res.json();
  }
}

/**
 * GitServiceClient - HTTP client wrapper for Python Git Service endpoints
 * Proxies git operations from Node.js to Python FastAPI
 */
export class GitServiceClient {
  constructor(options = {}) {
    this.pythonServiceUrl = options.pythonServiceUrl || PYTHON_SERVICE_URL;
    this.isInitialized = false;
  }

  async initialize() {
    if (this.isInitialized) return;

    try {
      const response = await fetch(`${this.pythonServiceUrl}/status`);
      if (!response.ok) {
        throw new Error(`Python service returned ${response.status}`);
      }
      this.isInitialized = true;
      console.log(`✅ GitServiceClient connected to Python service at ${this.pythonServiceUrl}`);
    } catch (error) {
      throw new Error(`Failed to connect to Python service: ${error.message}`);
    }
  }

  // ========================================================================
  // Repository Operations
  // ========================================================================

  /**
   * List all available git repositories
   */
  async listRepositories() {
    const res = await fetch(`${this.pythonServiceUrl}/git/repositories`);
    if (!res.ok) {
      throw new Error(`Failed to list repositories: ${await res.text()}`);
    }
    return await res.json();
  }

  /**
   * Get detailed info about a specific repository
   */
  async getRepositoryInfo(repoName) {
    const res = await fetch(`${this.pythonServiceUrl}/git/repositories/${encodeURIComponent(repoName)}/info`);
    if (!res.ok) {
      throw new Error(`Failed to get repository info: ${await res.text()}`);
    }
    return await res.json();
  }

  /**
   * Get recent commits from a repository
   */
  async getRepositoryCommits(repoName, limit = 10, noMerges = true) {
    const params = new URLSearchParams({
      limit: limit.toString(),
      no_merges: noMerges.toString()
    });
    const res = await fetch(`${this.pythonServiceUrl}/git/repositories/${encodeURIComponent(repoName)}/commits?${params}`);
    if (!res.ok) {
      throw new Error(`Failed to get commits: ${await res.text()}`);
    }
    return await res.json();
  }

  // ========================================================================
  // Pull Operations
  // ========================================================================

  /**
   * Pull a repository and optionally analyze changes
   */
  async pullRepository(repoName, options = {}) {
    const {
      analyzeChanges = true,
      maxFilesToAnalyze = 20,
      includeCodeAnalysis = true
    } = options;

    const res = await fetch(`${this.pythonServiceUrl}/git/pull`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        repo: repoName,
        analyze_changes: analyzeChanges,
        max_files_to_analyze: maxFilesToAnalyze,
        include_code_analysis: includeCodeAnalysis
      })
    });

    if (!res.ok) {
      throw new Error(`Failed to pull repository: ${await res.text()}`);
    }
    return await res.json();
  }

  /**
   * Pull all repositories (admin)
   */
  async pullAllRepositories() {
    const res = await fetch(`${this.pythonServiceUrl}/admin/git/pull-all`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    });

    if (!res.ok) {
      throw new Error(`Failed to pull all repositories: ${await res.text()}`);
    }
    return await res.json();
  }

  // ========================================================================
  // Analysis Operations
  // ========================================================================

  /**
   * Analyze commits in a date range
   */
  async analyzePastCommits(repoName, startDate, endDate) {
    const res = await fetch(`${this.pythonServiceUrl}/git/analyze-past`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        repo: repoName,
        start_date: startDate,
        end_date: endDate
      })
    });

    if (!res.ok) {
      throw new Error(`Failed to analyze commits: ${await res.text()}`);
    }
    return await res.json();
  }

  /**
   * Analyze the impact of a specific commit
   */
  async analyzeCommitImpact(repoName, commitHash) {
    const res = await fetch(`${this.pythonServiceUrl}/git/analyze-commit-impact`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        repo: repoName,
        commit_hash: commitHash
      })
    });

    if (!res.ok) {
      throw new Error(`Failed to analyze commit impact: ${await res.text()}`);
    }
    return await res.json();
  }

  /**
   * Get changed files between commits
   */
  async getChangedFiles(repoName, fromCommit, toCommit = 'HEAD') {
    const params = new URLSearchParams({
      from_commit: fromCommit,
      to_commit: toCommit
    });
    const res = await fetch(`${this.pythonServiceUrl}/git/changed-files/${encodeURIComponent(repoName)}?${params}`);
    if (!res.ok) {
      throw new Error(`Failed to get changed files: ${await res.text()}`);
    }
    return await res.json();
  }

  /**
   * Get file changes with status (Added/Modified/Deleted)
   */
  async getFileStatus(repoName, fromCommit = 'HEAD@{1}', toCommit = 'HEAD') {
    const params = new URLSearchParams({
      from_commit: fromCommit,
      to_commit: toCommit
    });
    const res = await fetch(`${this.pythonServiceUrl}/git/file-status/${encodeURIComponent(repoName)}?${params}`);
    if (!res.ok) {
      throw new Error(`Failed to get file status: ${await res.text()}`);
    }
    return await res.json();
  }

  // ========================================================================
  // Admin Operations
  // ========================================================================

  /**
   * Get all repositories with detailed status (admin)
   */
  async adminListRepositories() {
    const res = await fetch(`${this.pythonServiceUrl}/admin/git/repositories`);
    if (!res.ok) {
      throw new Error(`Failed to list admin repositories: ${await res.text()}`);
    }
    return await res.json();
  }
}

// Export singleton instances for convenience
export const sqlKnowledgeDB = new SQLKnowledgeDB();
export const documentationDB = new DocumentationDB();
export const codeContextDB = new CodeContextDB();
export const gitServiceClient = new GitServiceClient();
