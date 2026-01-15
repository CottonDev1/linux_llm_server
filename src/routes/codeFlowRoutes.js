/**
 * Code Flow Analysis Routes
 *
 * This module contains all code flow analysis endpoints for the RAG Server.
 * Routes are exported as a factory function that accepts dependencies via dependency injection.
 *
 * Extracted Routes:
 * - POST /api/code-flow - Complex multi-hop code flow queries (2 implementations)
 * - GET /api/method-lookup - Find methods by name, class, or signature (2 implementations)
 * - GET /api/call-chain - Build execution paths from UI events to database (1st implementation)
 * - POST /api/call-chain - Build execution paths from entry point (2nd implementation)
 *
 * Note: This file contains DUPLICATE route handlers from rag-server.js.
 * The first set (lines ~2181-2493) uses multi-stage retrieval and CodeFlowService.
 * The second set (lines ~4918-5082) uses AgenticRetrieval system.
 * These duplicates need to be resolved in rag-server.js refactoring.
 */

import express from 'express';

/**
 * Create code flow routes with injected dependencies
 *
 * @param {Object} deps - Dependencies object
 * @param {Object} deps.db - Vector database instance for vector search
 * @param {Object} deps.multiTableSearch - MultiTableSearch instance
 * @param {Object} deps.monitoring - MonitoringService instance
 * @param {Object} deps.queryService - QueryService instance (Phase 2)
 * @param {Object} deps.agenticRetrieval - AgenticRetrieval instance
 * @param {Object} deps.codeFlowService - CodeFlowService instance
 * @param {Map} deps.codeFlowCache - Cache for code flow queries
 * @param {number} deps.CACHE_TTL - Cache time-to-live in milliseconds
 * @returns {express.Router} Express router with code flow routes
 */
export default function createCodeFlowRoutes(deps) {
  const router = express.Router();

  const {
    db,
    multiTableSearch,
    monitoring,
    queryService,
    agenticRetrieval,
    codeFlowService,
    codeFlowCache,
    CACHE_TTL
  } = deps;

  // ============================================================================
  // FIRST IMPLEMENTATION: Multi-Stage Retrieval with CodeFlowService
  // ============================================================================

  /**
   * POST /api/code-flow (First Implementation)
   *
   * Complex multi-hop code flow queries using multi-stage retrieval
   *
   * Example: "How are bales committed to purchase subcontract in Gin?"
   *
   * Response includes:
   * - Business process documentation
   * - Key methods and their implementations
   * - UI entry points and event handlers
   * - Complete call chains from UI to database
   * - Data flow and database operations
   *
   * @route POST /api/code-flow
   * @deprecated This is the first implementation. Consider using the second implementation with AgenticRetrieval.
   */
  router.post('/api/code-flow-v1', async (req, res) => {
    try {
      const { query, project, maxHops = 5, includeCallGraph = true, detailed = false } = req.body;

      if (!query) {
        return res.status(400).json({
          success: false,
          error: 'Query is required'
        });
      }

      if (!project) {
        return res.status(400).json({
          success: false,
          error: 'Project scope is required (e.g., "gin", "warehouse")'
        });
      }

      console.log(`🔍 Code flow query: "${query}" in project: ${project}`);

      // Try Phase 2 Service Layer if available
      if (queryService) {
        try {
          const result = await queryService.executeCodeFlowQuery(query, {
            project,
            maxHops,
            detailed: includeCallGraph
          });

          res.json(result);
          return;
        } catch (serviceError) {
          monitoring.warn('Phase 2 QueryService code flow failed, falling back to direct implementation', { error: serviceError.message });
          // Fall through to original implementation
        }
      }

      // Check cache
      const cacheKey = `${project}:${query}:${includeCallGraph}`;
      const cached = codeFlowCache.get(cacheKey);
      if (cached && (Date.now() - cached.timestamp < CACHE_TTL)) {
        console.log(`✅ Cache hit for code flow query`);
        return res.json({ ...cached.data, cached: true });
      }

      // Stage 1: Classify query type and determine search strategy
      const queryClassification = await classifyCodeFlowQuery(query);
      console.log(`📋 Query type: ${queryClassification.type}`);

      // Stage 2: Execute multi-stage retrieval
      const retrievalResults = await executeMultiStageRetrieval(
        query,
        project,
        queryClassification,
        includeCallGraph
      );

      console.log(`📚 Retrieved: ${retrievalResults.totalResults} results across ${Object.keys(retrievalResults).length - 1} categories`);

      // Stage 3: Build call chains if requested
      let callChains = [];
      if (includeCallGraph && retrievalResults.methods.length > 0) {
        callChains = await buildCallChains(
          retrievalResults.methods,
          retrievalResults.uiEvents,
          project,
          maxHops
        );
        console.log(`🔗 Built ${callChains.length} call chains`);
      }

      // Stage 4: LLM synthesis
      const synthesizedAnswer = await synthesizeCodeFlowAnswer(
        query,
        retrievalResults,
        callChains,
        project
      );

      const response = {
        success: true,
        query,
        project,
        answer: synthesizedAnswer.answer,
        queryType: queryClassification.type,
        results: {
          businessProcesses: retrievalResults.businessProcesses.map(formatResult),
          methods: retrievalResults.methods.map(formatResult),
          classes: retrievalResults.classes.map(formatResult),
          uiEvents: retrievalResults.uiEvents.map(formatResult),
          callRelationships: retrievalResults.callRelationships.map(formatResult)
        },
        callChains: callChains,
        totalResults: retrievalResults.totalResults,
        processingTime: synthesizedAnswer.processingTime
      };

      // Cache the result
      codeFlowCache.set(cacheKey, {
        data: response,
        timestamp: Date.now()
      });

      res.json(response);

    } catch (error) {
      console.error('❌ Code flow query error:', error.message);
      res.status(500).json({
        success: false,
        error: 'Code flow query failed',
        details: error.message
      });
    }
  });

  /**
   * GET /api/method-lookup (First Implementation)
   *
   * Find methods by name, class, or signature
   *
   * Query params:
   * - method: Method name (partial match)
   * - class: Class name (partial match)
   * - signature: Method signature (partial match)
   * - project: Project scope
   * - limit: Max results (default 20)
   *
   * @route GET /api/method-lookup
   * @deprecated This is the first implementation. Consider consolidating with the second implementation.
   */
  router.get('/api/method-lookup-v1', async (req, res) => {
    try {
      const { method, class: className, signature, project, limit = 20 } = req.query;

      if (!method && !className && !signature) {
        return res.status(400).json({
          success: false,
          error: 'At least one search parameter required: method, class, or signature'
        });
      }

      console.log(`🔎 Method lookup: method=${method}, class=${className}, signature=${signature}, project=${project}`);

      // Build search query
      const searchTerms = [];
      if (method) searchTerms.push(`method ${method}`);
      if (className) searchTerms.push(`class ${className}`);
      if (signature) searchTerms.push(`signature ${signature}`);

      const searchQuery = searchTerms.join(' ');

      // Search code_methods table
      const searchOptions = {
        limit: parseInt(limit),
        filter: { category: 'code', type: 'method' }
      };

      if (project) {
        searchOptions.project = project;
      }

      const results = await db.search(searchQuery, searchOptions);

      // Filter results by metadata if specific fields were requested
      let filteredResults = results;

      if (method) {
        filteredResults = filteredResults.filter(r =>
          r.metadata?.methodName?.toLowerCase().includes(method.toLowerCase())
        );
      }

      if (className) {
        filteredResults = filteredResults.filter(r =>
          r.metadata?.className?.toLowerCase().includes(className.toLowerCase())
        );
      }

      if (signature) {
        filteredResults = filteredResults.filter(r =>
          r.metadata?.signature?.toLowerCase().includes(signature.toLowerCase())
        );
      }

      console.log(`✅ Found ${filteredResults.length} methods`);

      res.json({
        success: true,
        results: filteredResults.map(r => ({
          methodName: r.metadata?.methodName,
          className: r.metadata?.className,
          namespace: r.metadata?.namespace,
          signature: r.metadata?.signature,
          filePath: r.metadata?.filePath,
          startLine: r.metadata?.startLine,
          endLine: r.metadata?.endLine,
          returnType: r.metadata?.returnType,
          isPublic: r.metadata?.isPublic,
          isStatic: r.metadata?.isStatic,
          isAsync: r.metadata?.isAsync,
          purposeSummary: r.metadata?.purposeSummary,
          callsMethods: tryParseJSON(r.metadata?.callsMethod),
          calledByMethods: tryParseJSON(r.metadata?.calledByMethod),
          databaseTables: tryParseJSON(r.metadata?.databaseTables),
          businessDomain: r.metadata?.businessDomain,
          similarity: r.similarity,
          project: r.metadata?.project
        })),
        totalResults: filteredResults.length
      });

    } catch (error) {
      console.error('❌ Method lookup error:', error.message);
      res.status(500).json({
        success: false,
        error: 'Method lookup failed',
        details: error.message
      });
    }
  });

  /**
   * GET /api/call-chain (First Implementation)
   *
   * Build execution paths from UI events to database operations
   *
   * Query params:
   * - startMethod: Starting point (e.g., "btnCommit_Click")
   * - endMethod: Target method (optional)
   * - eventHandler: UI event handler name (optional)
   * - project: Project scope (required)
   * - maxDepth: Maximum call chain depth (default 10)
   *
   * @route GET /api/call-chain
   * @deprecated This is the first implementation using GET. Consider consolidating with the POST implementation.
   */
  router.get('/api/call-chain-v1', async (req, res) => {
    try {
      const {
        startMethod,
        endMethod,
        eventHandler,
        project,
        maxDepth = 10
      } = req.query;

      if (!project) {
        return res.status(400).json({
          success: false,
          error: 'Project scope is required'
        });
      }

      if (!startMethod && !eventHandler) {
        return res.status(400).json({
          success: false,
          error: 'Either startMethod or eventHandler is required'
        });
      }

      console.log(`🔗 Building call chain from ${startMethod || eventHandler} in ${project}`);

      // Find starting point
      let startingMethods = [];

      if (eventHandler) {
        // Search for UI event handlers
        const eventResults = await db.search(`event handler ${eventHandler}`, {
          project,
          filter: { category: 'ui-mapping' },
          limit: 5
        });

        startingMethods = eventResults.map(e => e.metadata?.handlerMethod).filter(Boolean);
      } else {
        startingMethods = [startMethod];
      }

      if (startingMethods.length === 0) {
        return res.json({
          success: true,
          callChains: [],
          message: 'No starting methods found'
        });
      }

      // Build call chains for each starting method
      const allChains = [];

      for (const methodName of startingMethods) {
        const chains = await traverseCallGraph(
          methodName,
          project,
          endMethod,
          parseInt(maxDepth)
        );
        allChains.push(...chains);
      }

      console.log(`✅ Found ${allChains.length} call chains`);

      res.json({
        success: true,
        startMethod: startMethod || startingMethods[0],
        endMethod: endMethod || 'any',
        project,
        callChains: allChains,
        totalChains: allChains.length
      });

    } catch (error) {
      console.error('❌ Call chain error:', error.message);
      res.status(500).json({
        success: false,
        error: 'Call chain construction failed',
        details: error.message
      });
    }
  });

  // ============================================================================
  // SECOND IMPLEMENTATION: Agentic Retrieval System
  // ============================================================================

  /**
   * POST /api/code-flow (Second Implementation - ACTIVE)
   *
   * Complex code flow analysis endpoint using agentic retrieval
   *
   * Answers questions like:
   * - "How are bales committed to a purchase subcontract in Gin?"
   * - "What's the execution path from UI to database for load creation?"
   * - "Show me the complete flow for processing inbound files"
   *
   * Uses multi-hop retrieval to build comprehensive answers
   *
   * @route POST /api/code-flow
   */
  router.post('/api/code-flow', async (req, res) => {
    try {
      const { question, project } = req.body;

      if (!question) {
        return res.status(400).json({
          success: false,
          error: 'Question is required'
        });
      }

      console.log(`\n${'='.repeat(80)}`);
      console.log(`🤖 CODE FLOW QUERY: "${question}"`);
      console.log(`${'='.repeat(80)}\n`);

      const startTime = Date.now();

      // Execute agentic retrieval
      const result = await agenticRetrieval.answerQuestion(question, project);

      const duration = Date.now() - startTime;

      console.log(`\n✅ Code flow analysis complete (${duration}ms)`);

      res.json({
        success: result.success,
        question,
        answer: result.answer,
        confidence: result.confidence,
        sources: result.sources,
        reasoning: result.reasoning,
        metadata: {
          ...result.metadata,
          duration
        }
      });

    } catch (error) {
      console.error('❌ Code flow analysis failed:', error.message);
      res.status(500).json({
        success: false,
        error: 'Code flow analysis failed',
        details: error.message
      });
    }
  });

  /**
   * GET /api/method-lookup (Second Implementation - ACTIVE)
   *
   * Find methods by name, class, or signature
   * Returns semantic descriptions and call graph info
   *
   * @route GET /api/method-lookup
   */
  router.get('/api/method-lookup', async (req, res) => {
    try {
      const { methodName, className, project, limit = 10 } = req.query;

      if (!methodName) {
        return res.status(400).json({
          success: false,
          error: 'methodName parameter is required'
        });
      }

      console.log(`🔍 Method lookup: ${className ? className + '.' : ''}${methodName}`);

      // Build search query
      let searchQuery = methodName;
      if (className) {
        searchQuery = `${className} ${methodName}`;
      }

      // Search code_methods table
      const results = await db.search(searchQuery, {
        project,
        filter: { category: 'code', type: 'method' },
        limit: parseInt(limit)
      });

      // Filter by exact method name if needed
      const filtered = results.filter(r => {
        const metadata = r.metadata || {};
        if (metadata.methodName?.toLowerCase() === methodName.toLowerCase()) {
          if (className) {
            return metadata.className?.toLowerCase() === className.toLowerCase();
          }
          return true;
        }
        return false;
      });

      console.log(`✅ Found ${filtered.length} matching methods`);

      res.json({
        success: true,
        methods: filtered.map(m => ({
          methodName: m.metadata?.methodName,
          className: m.metadata?.className,
          namespace: m.metadata?.namespace,
          signature: m.metadata?.signature || m.metadata?.fullMethodSignature,
          purposeSummary: m.metadata?.purposeSummary,
          filePath: m.metadata?.filePath,
          similarity: m.similarity,
          description: m.content.substring(0, 500)
        })),
        totalResults: filtered.length
      });

    } catch (error) {
      console.error('❌ Method lookup failed:', error.message);
      res.status(500).json({
        success: false,
        error: 'Method lookup failed',
        details: error.message
      });
    }
  });

  /**
   * POST /api/call-chain (Second Implementation - ACTIVE)
   *
   * Build execution path from entry point to database operations
   * Example: UI button click → Event handler → Service method → Data layer → SQL
   *
   * @route POST /api/call-chain
   */
  router.post('/api/call-chain', async (req, res) => {
    try {
      const { entryPoint, project, maxDepth = 10 } = req.body;

      if (!entryPoint) {
        return res.status(400).json({
          success: false,
          error: 'entryPoint is required (e.g., "BaleCommitmentWindow.btnCommit_Click")'
        });
      }

      console.log(`🔗 Building call chain from: ${entryPoint}`);

      // Search for call graph entries starting from entry point
      const callGraphResults = await db.search(entryPoint, {
        project,
        filter: { category: 'relationship', type: 'method-call' },
        limit: 50
      });

      // Build call tree
      const callTree = buildCallTree(entryPoint, callGraphResults, maxDepth);

      console.log(`✅ Built call tree with ${callTree.totalMethods} methods`);

      res.json({
        success: true,
        entryPoint,
        callTree,
        rawResults: callGraphResults.length
      });

    } catch (error) {
      console.error('❌ Call chain failed:', error.message);
      res.status(500).json({
        success: false,
        error: 'Call chain failed',
        details: error.message
      });
    }
  });

  // ============================================================================
  // HELPER FUNCTIONS (from rag-server.js)
  // ============================================================================

  /**
   * Classify the type of code flow query
   */
  async function classifyCodeFlowQuery(query) {
    const lowerQuery = query.toLowerCase();

    // Pattern matching for query classification
    if (lowerQuery.match(/how .*(process|workflow|work|happen)/)) {
      return { type: 'business-process', confidence: 0.9 };
    }

    if (lowerQuery.match(/(what|which) methods? .*(update|insert|delete|modify|change)/)) {
      return { type: 'method-search-by-action', confidence: 0.85 };
    }

    if (lowerQuery.match(/call (chain|graph|tree|path|flow)/)) {
      return { type: 'call-chain', confidence: 0.9 };
    }

    if (lowerQuery.match(/(ui|button|click|event|user)/)) {
      return { type: 'ui-interaction', confidence: 0.8 };
    }

    if (lowerQuery.match(/class .*(do|handle|manage|responsible)/)) {
      return { type: 'class-responsibility', confidence: 0.8 };
    }

    if (lowerQuery.match(/database|table|sql|query/)) {
      return { type: 'data-operation', confidence: 0.85 };
    }

    return { type: 'general', confidence: 0.5 };
  }

  /**
   * Execute multi-stage retrieval for code flow
   */
  async function executeMultiStageRetrieval(query, project, classification, includeCallGraph) {
    const results = {
      businessProcesses: [],
      methods: [],
      classes: [],
      uiEvents: [],
      callRelationships: [],
      totalResults: 0
    };

    // Determine which stages to execute based on query type
    const stages = determineRetrievalStages(classification.type);

    // Execute searches in parallel
    const searchPromises = [];

    // Stage 1: Business processes
    if (stages.includes('business-process')) {
      searchPromises.push(
        db.search(query, {
          project,
          filter: { category: 'business-process' },
          limit: 3
        }).then(r => { results.businessProcesses = r; })
      );
    }

    // Stage 2: Methods
    if (stages.includes('methods')) {
      searchPromises.push(
        db.search(query, {
          project,
          filter: { category: 'code', type: 'method' },
          limit: 15
        }).then(r => { results.methods = r; })
      );
    }

    // Stage 3: Classes
    if (stages.includes('classes')) {
      searchPromises.push(
        db.search(query, {
          project,
          filter: { category: 'code', type: 'class' },
          limit: 10
        }).then(r => { results.classes = r; })
      );
    }

    // Stage 4: UI events
    if (stages.includes('ui-events')) {
      searchPromises.push(
        db.search(query, {
          project,
          filter: { category: 'ui-mapping' },
          limit: 5
        }).then(r => { results.uiEvents = r; })
      );
    }

    // Stage 5: Call relationships
    if (includeCallGraph && stages.includes('call-graph')) {
      searchPromises.push(
        db.search(query, {
          project,
          filter: { category: 'relationship', type: 'method-call' },
          limit: 20
        }).then(r => { results.callRelationships = r; })
      );
    }

    // Wait for all searches to complete
    await Promise.all(searchPromises);

    // Calculate total results
    results.totalResults =
      results.businessProcesses.length +
      results.methods.length +
      results.classes.length +
      results.uiEvents.length +
      results.callRelationships.length;

    return results;
  }

  /**
   * Determine which retrieval stages to execute based on query type
   */
  function determineRetrievalStages(queryType) {
    const stageMap = {
      'business-process': ['business-process', 'methods', 'ui-events', 'call-graph'],
      'method-search-by-action': ['methods', 'call-graph'],
      'call-chain': ['methods', 'call-graph', 'ui-events'],
      'ui-interaction': ['ui-events', 'methods', 'call-graph'],
      'class-responsibility': ['classes', 'methods'],
      'data-operation': ['methods', 'call-graph'],
      'general': ['business-process', 'methods', 'classes', 'ui-events']
    };

    return stageMap[queryType] || ['methods'];
  }

  /**
   * Build call chains from retrieved methods
   */
  async function buildCallChains(methods, uiEvents, project, maxHops) {
    const chains = [];

    // Start from UI events if available
    const startingPoints = uiEvents.length > 0
      ? uiEvents.map(e => e.metadata?.handlerMethod).filter(Boolean)
      : methods.slice(0, 3).map(m => m.metadata?.methodName).filter(Boolean);

    for (const startMethod of startingPoints) {
      try {
        const methodChains = await traverseCallGraph(startMethod, project, null, maxHops);
        chains.push(...methodChains);
      } catch (error) {
        console.warn(`⚠️ Failed to build chain for ${startMethod}:`, error.message);
      }
    }

    return chains;
  }

  /**
   * Traverse call graph starting from a method
   */
  async function traverseCallGraph(startMethod, project, targetMethod, maxDepth) {
    const visited = new Set();
    const chains = [];

    async function traverse(currentMethod, currentChain, depth) {
      if (depth > maxDepth) return;
      if (visited.has(currentMethod)) return; // Prevent cycles

      visited.add(currentMethod);
      currentChain.push({ method: currentMethod, depth });

      // Check if we reached target (if specified)
      if (targetMethod && currentMethod.includes(targetMethod)) {
        chains.push([...currentChain]);
        currentChain.pop();
        return;
      }

      // Find methods called by current method
      const methodInfo = await db.search(`method ${currentMethod}`, {
        project,
        filter: { category: 'code', type: 'method' },
        limit: 1
      });

      if (methodInfo.length > 0) {
        const calledMethods = tryParseJSON(methodInfo[0].metadata?.callsMethod) || [];
        const databaseTables = tryParseJSON(methodInfo[0].metadata?.databaseTables) || [];

        // If this method touches database, consider it a terminal node
        if (databaseTables.length > 0) {
          currentChain[currentChain.length - 1].databaseTables = databaseTables;
          chains.push([...currentChain]);
        } else if (calledMethods.length > 0) {
          // Recursively traverse called methods
          for (const calledMethod of calledMethods.slice(0, 3)) { // Limit to 3 to prevent explosion
            await traverse(calledMethod, currentChain, depth + 1);
          }
        } else {
          // Leaf node - add to chains
          chains.push([...currentChain]);
        }
      }

      currentChain.pop();
    }

    await traverse(startMethod, [], 0);

    return chains.map(chain => ({
      startMethod: chain[0].method,
      endMethod: chain[chain.length - 1].method,
      steps: chain,
      depth: chain.length,
      touchesDatabase: chain.some(step => step.databaseTables)
    }));
  }

  /**
   * Synthesize final answer using LLM
   */
  async function synthesizeCodeFlowAnswer(query, results, callChains, project) {
    const startTime = Date.now();

    // Build context for LLM
    const contextSections = [];

    // Business processes
    if (results.businessProcesses.length > 0) {
      contextSections.push(`## Business Processes\n\n` +
        results.businessProcesses.map(p => p.content).join('\n\n---\n\n'));
    }

    // Key methods
    if (results.methods.length > 0) {
      contextSections.push(`## Key Methods\n\n` +
        results.methods.map(m => `### ${m.metadata?.methodName}\n${m.content.substring(0, 500)}...`).join('\n\n'));
    }

    // UI entry points
    if (results.uiEvents.length > 0) {
      contextSections.push(`## UI Entry Points\n\n` +
        results.uiEvents.map(e => e.content).join('\n\n'));
    }

    // Call chains
    if (callChains.length > 0) {
      contextSections.push(`## Call Chains\n\n` +
        callChains.slice(0, 5).map((chain, idx) =>
          `Chain ${idx + 1}: ${chain.steps.map(s => s.method).join(' → ')}`
        ).join('\n'));
    }

    const context = contextSections.join('\n\n');

    // Create LLM prompt
    const systemPrompt = `You are a code analysis expert. Explain code flow and architecture based on the provided analysis.
Focus on answering the user's question with:
1. User workflow (how users interact with the feature)
2. Code flow (execution path from UI to database)
3. Business logic (validations, rules, data transformations)
4. Data operations (database tables affected, operations performed)
5. Error handling and edge cases

Be concise but comprehensive. Use technical terminology appropriately.`;

    const userPrompt = `Question: ${query}

Project: ${project}

Code Analysis Results:
${context}

Please provide a clear, structured answer to the question.`;

    // Query LLM via Python service
    try {
      const pythonUrl = process.env.PYTHON_SERVICE_URL || 'http://localhost:8001';
      const llmResponse = await fetch(`${pythonUrl}/llm/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: userPrompt,
          system: systemPrompt || 'You are a helpful code analysis assistant.',
          max_tokens: 2048,
          temperature: 0.5,
          use_cache: true
        })
      });

      if (!llmResponse.ok) {
        throw new Error(`LLM API error: ${llmResponse.statusText}`);
      }

      const llmData = await llmResponse.json();

      if (!llmData.success) {
        throw new Error(llmData.error || 'LLM generation failed');
      }

      return {
        answer: llmData.response || '',
        processingTime: Date.now() - startTime
      };
    } catch (error) {
      console.error('❌ LLM synthesis failed:', error.message);

      // Fallback: Return structured summary without LLM
      return {
        answer: generateFallbackAnswer(query, results, callChains),
        processingTime: Date.now() - startTime,
        fallback: true
      };
    }
  }

  /**
   * Generate fallback answer without LLM
   */
  function generateFallbackAnswer(query, results, callChains) {
    const sections = [];

    sections.push(`## Answer to: "${query}"\n`);

    if (results.businessProcesses.length > 0) {
      sections.push(`### Business Process Overview\n\n${results.businessProcesses[0].content.substring(0, 500)}...\n`);
    }

    if (results.methods.length > 0) {
      sections.push(`### Key Methods Found\n\n` +
        results.methods.slice(0, 5).map(m =>
          `- **${m.metadata?.className}.${m.metadata?.methodName}**: ${m.metadata?.purposeSummary || 'Method implementation'}`
        ).join('\n'));
    }

    if (callChains.length > 0) {
      sections.push(`\n### Execution Flow\n\n` +
        callChains.slice(0, 3).map((chain, idx) =>
          `**Chain ${idx + 1}:**\n` +
          chain.steps.map((step, i) => `${i + 1}. ${step.method}`).join('\n')
        ).join('\n\n'));
    }

    return sections.join('\n\n');
  }

  /**
   * Helper function to build call tree from call graph results
   */
  function buildCallTree(rootMethod, callGraphResults, maxDepth, depth = 0, visited = new Set()) {
    if (depth >= maxDepth || visited.has(rootMethod)) {
      return null;
    }

    visited.add(rootMethod);

    // Find calls from this method
    const outgoingCalls = callGraphResults.filter(r => {
      const caller = r.metadata?.callerMethod;
      return caller && caller.includes(rootMethod);
    });

    const children = outgoingCalls
      .map(call => {
        const callee = call.metadata?.calleeMethod;
        if (!callee) return null;

        return {
          method: callee,
          caller: rootMethod,
          callType: call.metadata?.callType || 'direct',
          children: buildCallTree(callee, callGraphResults, maxDepth, depth + 1, new Set(visited))
        };
      })
      .filter(Boolean);

    return {
      method: rootMethod,
      depth,
      children,
      totalMethods: 1 + children.reduce((sum, c) => sum + (c.totalMethods || 0), 0)
    };
  }

  /**
   * Format search result for response
   */
  function formatResult(result) {
    return {
      id: result.id,
      similarity: result.similarity,
      content: result.content?.substring(0, 300),
      metadata: result.metadata,
      project: result.metadata?.project,
      category: result.metadata?.category,
      type: result.metadata?.type
    };
  }

  /**
   * Safely parse JSON strings
   */
  function tryParseJSON(jsonString) {
    if (!jsonString) return [];
    if (typeof jsonString !== 'string') return jsonString;

    try {
      return JSON.parse(jsonString);
    } catch {
      return [];
    }
  }

  return router;
}
