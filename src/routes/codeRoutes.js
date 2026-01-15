/**
 * Code Assistance Routes Module
 * Handles all code assistance query endpoints for C# codebase Q&A
 *
 * All code data is retrieved from MongoDB via the Python service.
 *
 * Routes:
 * - POST /query - Main code assistance query endpoint with RAG
 * - POST /query/stream - Streaming code assistance with SSE
 * - POST /feedback - Submit feedback on a response
 * - GET /stats - Get code assistance statistics
 */

import express from 'express';
import fetch from 'node-fetch';
import { v4 as uuidv4 } from 'uuid';

const router = express.Router();

/**
 * Create code assistance routes with dependency injection
 * @param {Object} deps - Dependencies needed by routes
 * @param {Object} deps.monitoring - Monitoring instance
 * @param {string} deps.pythonServiceUrl - Python service URL
 * @returns {express.Router} Express router
 */
export default function createCodeRoutes(deps) {
  const {
    monitoring,
    pythonServiceUrl = 'http://localhost:8001'
  } = deps;

  /**
   * POST /api/code/query
   * Main code assistance endpoint with RAG: code retrieval + LLM generation
   */
  router.post('/query', async (req, res) => {
    const startTime = Date.now();
    const responseId = uuidv4();

    try {
      const {
        query,
        project,
        model = 'deepseek-coder-v2:16b',
        include_call_chains = true,
        max_depth = 2,
        history = []
      } = req.body;

      if (!query) {
        return res.status(400).json({ error: 'Query is required' });
      }

      monitoring.info(`Code assistance query: "${query}"`, { project, model, responseId });

      // Step 1: Retrieve relevant code context from Python/MongoDB
      monitoring.info('Retrieving code context from MongoDB...');
      const retrievalStartTime = Date.now();

      let methods = [];
      let classes = [];
      let eventHandlers = [];
      let callChain = [];

      try {
        // Search methods
        const methodsResponse = await fetch(`${pythonServiceUrl}/roslyn/search/methods?query=${encodeURIComponent(query)}&project=${project || ''}&limit=10`);
        if (methodsResponse.ok) {
          const methodsData = await methodsResponse.json();
          methods = methodsData.results || [];
        }

        // Search classes
        const classesResponse = await fetch(`${pythonServiceUrl}/roslyn/search/classes?query=${encodeURIComponent(query)}&project=${project || ''}&limit=5`);
        if (classesResponse.ok) {
          const classesData = await classesResponse.json();
          classes = classesData.results || [];
        }

        // Search event handlers (for UI-related queries)
        if (query.toLowerCase().includes('click') || query.toLowerCase().includes('button') || query.toLowerCase().includes('event')) {
          const eventsResponse = await fetch(`${pythonServiceUrl}/roslyn/search/event-handlers?query=${encodeURIComponent(query)}&project=${project || ''}&limit=5`);
          if (eventsResponse.ok) {
            const eventsData = await eventsResponse.json();
            eventHandlers = eventsData.results || [];
          }
        }

        // Get call chain for top methods if requested
        if (include_call_chains && methods.length > 0) {
          const topMethod = methods[0];
          if (topMethod.method_name && topMethod.class_name) {
            const callChainResponse = await fetch(
              `${pythonServiceUrl}/roslyn/call-chain?method_name=${encodeURIComponent(topMethod.method_name)}&class_name=${encodeURIComponent(topMethod.class_name)}&project=${project || ''}&direction=both&max_depth=${max_depth}`
            );
            if (callChainResponse.ok) {
              const callChainData = await callChainResponse.json();
              // Build call chain string
              const callers = (callChainData.callers || []).map(c => `${c.class}.${c.method}`);
              const callees = (callChainData.callees || []).map(c => `${c.class}.${c.method}`);
              callChain = [...callers.slice(0, 3), `${topMethod.class_name}.${topMethod.method_name}`, ...callees.slice(0, 3)];
            }
          }
        }
      } catch (retrievalError) {
        monitoring.error('Code retrieval failed', retrievalError);
        // Continue with empty results
      }

      const retrievalTime = Date.now() - retrievalStartTime;
      monitoring.info(`Code retrieval completed in ${retrievalTime}ms`, {
        methods: methods.length,
        classes: classes.length,
        eventHandlers: eventHandlers.length
      });

      // Step 2: Build context for LLM
      const contextParts = [];
      const sources = [];

      // Add methods to context
      for (const method of methods.slice(0, 8)) {
        const methodInfo = `
Method: ${method.class_name || ''}.${method.method_name || ''}
File: ${method.file_path || 'Unknown'}:${method.line_number || 0}
Signature: ${method.signature || ''}
Summary: ${method.summary || 'No summary available'}
${method.sql_calls && method.sql_calls.length > 0 ? `SQL Operations: ${method.sql_calls.length} database calls` : ''}
`;
        contextParts.push(methodInfo);

        sources.push({
          type: 'method',
          name: `${method.class_name || ''}.${method.method_name || ''}`,
          file: method.file_path || '',
          line: method.line_number || 0,
          similarity: method.similarity || 0,
          snippet: method.signature || ''
        });
      }

      // Add classes to context
      for (const cls of classes.slice(0, 4)) {
        const classInfo = `
Class: ${cls.namespace || ''}.${cls.class_name || ''}
File: ${cls.file_path || 'Unknown'}
Base Class: ${cls.base_class || 'None'}
Interfaces: ${(cls.interfaces || []).join(', ') || 'None'}
Methods: ${(cls.methods || []).slice(0, 5).join(', ') || 'None'}
`;
        contextParts.push(classInfo);

        sources.push({
          type: 'class',
          name: `${cls.namespace || ''}.${cls.class_name || ''}`,
          file: cls.file_path || '',
          line: 0,
          similarity: cls.similarity || 0,
          snippet: `class ${cls.class_name}${cls.base_class ? ` : ${cls.base_class}` : ''}`
        });
      }

      // Add event handlers if found
      for (const handler of eventHandlers.slice(0, 3)) {
        const handlerInfo = `
Event Handler: ${handler.event_name || ''} -> ${handler.handler_method || ''}
UI Element: ${handler.element_name || ''} (${handler.ui_element_type || ''})
Handler Class: ${handler.handler_class || ''}
`;
        contextParts.push(handlerInfo);

        sources.push({
          type: 'event_handler',
          name: `${handler.event_name} -> ${handler.handler_method}`,
          file: handler.file_path || '',
          line: handler.line_number || 0,
          similarity: handler.similarity || 0,
          snippet: `${handler.element_name}.${handler.event_name} += ${handler.handler_method}`
        });
      }

      // Add call chain if available
      if (callChain.length > 0) {
        contextParts.push(`\nCall Flow:\n${callChain.join(' -> ')}`);
      }

      const contextStr = contextParts.join('\n---\n');

      // Step 3: Generate response with LLM
      monitoring.info(`Generating response with model ${model}...`);
      const generationStartTime = Date.now();

      // Build conversation history for context
      let conversationContext = '';
      if (history && history.length > 0) {
        const recentHistory = history.slice(-4); // Last 4 messages
        conversationContext = recentHistory.map(msg =>
          `${msg.role === 'user' ? 'User' : 'Assistant'}: ${msg.content}`
        ).join('\n\n');
        conversationContext = `\nPrevious conversation:\n${conversationContext}\n\n`;
      }

      const prompt = `You are a helpful code assistant that explains C# code to developers.
Based on the following code context, answer the user's question.
Always cite specific methods, classes, and file paths in your answer.
If you're not sure about something, say so.
${conversationContext}
CODE CONTEXT:
${contextStr}

USER QUESTION: ${query}

Provide a clear, concise answer that:
1. Directly addresses the question
2. References specific code elements (ClassName.MethodName format)
3. Explains the flow if relevant
4. Mentions file paths for key code

ANSWER:`;

      let answer = '';
      let tokenUsage = { promptTokens: 0, completionTokens: 0, totalTokens: 0 };

      try {
        // Use Python LLM service for code generation
        const llmResponse = await fetch(`${pythonServiceUrl}/llm/generate`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            prompt: prompt,
            system: 'You are a helpful code assistant that answers questions about code.',
            max_tokens: 1000,
            temperature: 0.3,
            use_cache: true
          }),
          signal: AbortSignal.timeout(120000)
        });

        if (llmResponse.ok) {
          const llmData = await llmResponse.json();
          if (llmData.success) {
            answer = llmData.response || '';
            tokenUsage = {
              promptTokens: llmData.token_usage?.prompt_tokens || 0,
              completionTokens: llmData.token_usage?.completion_tokens || llmData.token_usage?.response_tokens || 0,
              totalTokens: llmData.token_usage?.total_tokens || 0
            };
          } else {
            throw new Error(llmData.error || 'LLM generation failed');
          }
        } else {
          const errorText = await llmResponse.text();
          throw new Error(`LLM error: ${llmResponse.status} - ${errorText}`);
        }
      } catch (llmError) {
        monitoring.error('LLM generation failed', llmError);
        answer = 'I encountered an error generating the response. Please try again or rephrase your question.';
      }

      const generationTime = Date.now() - generationStartTime;
      const totalTime = Date.now() - startTime;

      monitoring.info(`Response generated in ${generationTime}ms, total time ${totalTime}ms`);

      // Step 4: Log interaction for feedback
      try {
        await fetch(`${pythonServiceUrl}/code-context`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            id: responseId,
            content: JSON.stringify({
              query,
              answer: answer.substring(0, 5000),
              sources: sources.map(s => s.name),
              call_chain: callChain
            }),
            metadata: {
              type: 'code_interaction',
              response_id: responseId,
              project: project || 'all',
              model_used: model,
              retrieval_time_ms: retrievalTime,
              generation_time_ms: generationTime,
              total_time_ms: totalTime,
              feedback_received: false
            }
          })
        });
      } catch (logError) {
        monitoring.error('Failed to log interaction', logError);
        // Continue anyway
      }

      // Return response
      res.json({
        answer,
        sources,
        call_chain: callChain,
        response_id: responseId,
        tokenUsage,
        timing: {
          retrieval_ms: retrievalTime,
          generation_ms: generationTime,
          total_ms: totalTime
        }
      });

    } catch (error) {
      monitoring.error('Code assistance query failed', error, { responseId });
      res.status(500).json({
        error: error.message,
        response_id: responseId
      });
    }
  });

  /**
   * POST /api/code/query/stream
   * Streaming code assistance with Server-Sent Events
   */
  router.post('/query/stream', async (req, res) => {
    const responseId = uuidv4();
    const startTime = Date.now();

    // Set up SSE
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');

    try {
      const {
        query,
        project,
        model = 'deepseek-coder-v2:16b',
        include_call_chains = true,
        max_depth = 2
      } = req.body;

      if (!query) {
        res.write(`data: ${JSON.stringify({ error: 'Query is required' })}\n\n`);
        res.end();
        return;
      }

      monitoring.info(`Streaming code query: "${query}"`, { project, model, responseId });

      // Send initial status
      res.write(`data: ${JSON.stringify({ status: 'retrieving', message: 'Searching codebase...' })}\n\n`);

      // Retrieve code context (same as non-streaming)
      let methods = [];
      let classes = [];
      let callChain = [];
      const sources = [];

      try {
        const methodsResponse = await fetch(`${pythonServiceUrl}/roslyn/search/methods?query=${encodeURIComponent(query)}&project=${project || ''}&limit=10`);
        if (methodsResponse.ok) {
          const methodsData = await methodsResponse.json();
          methods = methodsData.results || [];
        }

        const classesResponse = await fetch(`${pythonServiceUrl}/roslyn/search/classes?query=${encodeURIComponent(query)}&project=${project || ''}&limit=5`);
        if (classesResponse.ok) {
          const classesData = await classesResponse.json();
          classes = classesData.results || [];
        }
      } catch (err) {
        monitoring.error('Streaming retrieval error', err);
      }

      // Build sources
      for (const method of methods.slice(0, 8)) {
        sources.push({
          type: 'method',
          name: `${method.class_name || ''}.${method.method_name || ''}`,
          file: method.file_path || '',
          line: method.line_number || 0,
          similarity: method.similarity || 0
        });
      }

      for (const cls of classes.slice(0, 4)) {
        sources.push({
          type: 'class',
          name: `${cls.namespace || ''}.${cls.class_name || ''}`,
          file: cls.file_path || '',
          line: 0,
          similarity: cls.similarity || 0
        });
      }

      // Send sources
      res.write(`data: ${JSON.stringify({ status: 'sources', sources, call_chain: callChain, response_id: responseId })}\n\n`);

      // Build context and prompt
      const contextParts = methods.slice(0, 8).map(m =>
        `Method: ${m.class_name}.${m.method_name}\nFile: ${m.file_path}:${m.line_number}`
      );
      const contextStr = contextParts.join('\n---\n');

      const prompt = `You are a helpful code assistant explaining C# code.
Based on the code context below, answer the question.

CODE CONTEXT:
${contextStr}

USER QUESTION: ${query}

ANSWER:`;

      // Stream from Python LLM service
      res.write(`data: ${JSON.stringify({ status: 'generating', message: 'Generating response...' })}\n\n`);

      try {
        // Use Python LLM service streaming endpoint
        const llmResponse = await fetch(`${pythonServiceUrl}/llm/generate-stream`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            prompt: prompt,
            system: 'You are a helpful code assistant that answers questions about code.',
            max_tokens: 1000,
            temperature: 0.3
          })
        });

        if (!llmResponse.ok) {
          throw new Error(`LLM error: ${llmResponse.status}`);
        }

        // Stream the response - Python SSE format
        let fullAnswer = '';
        const reader = llmResponse.body;

        for await (const chunk of reader) {
          const lines = chunk.toString().split('\n').filter(line => line.trim());

          for (const line of lines) {
            if (!line.startsWith('data:')) continue;
            const dataStr = line.slice(5).trim();
            try {
              const data = JSON.parse(dataStr);
              if (data.content) {
                fullAnswer += data.content;
                res.write(`data: ${JSON.stringify({ status: 'streaming', token: data.content })}\n\n`);
              }
              if (data.done) {
                const totalTime = Date.now() - startTime;
                res.write(`data: ${JSON.stringify({
                  status: 'complete',
                  answer: fullAnswer,
                  response_id: responseId,
                  timing: { total_ms: totalTime }
                })}\n\n`);
              }
              if (data.error) {
                throw new Error(data.error);
              }
            } catch (parseError) {
              // Skip invalid JSON lines
            }
          }
        }
      } catch (streamError) {
        monitoring.error('Streaming generation error', streamError);
        res.write(`data: ${JSON.stringify({ status: 'error', error: streamError.message })}\n\n`);
      }

      res.end();

    } catch (error) {
      monitoring.error('Streaming code query failed', error);
      res.write(`data: ${JSON.stringify({ status: 'error', error: error.message })}\n\n`);
      res.end();
    }
  });

  /**
   * POST /api/code/feedback
   * Submit feedback on a code assistance response
   */
  router.post('/feedback', async (req, res) => {
    try {
      const {
        response_id,
        is_helpful,
        error_category,
        comment,
        expected_methods
      } = req.body;

      if (!response_id) {
        return res.status(400).json({ error: 'response_id is required' });
      }

      monitoring.info(`Code feedback received`, { response_id, is_helpful, error_category });

      // Forward to Python service
      const feedbackResponse = await fetch(`${pythonServiceUrl}/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          feedback_type: is_helpful ? 'rating' : 'correction',
          query_id: response_id,
          rating: {
            is_helpful,
            score: is_helpful ? 4 : 1,
            comment: comment || null
          },
          correction: !is_helpful ? {
            error_type: error_category || 'other',
            comment: comment || null,
            expected_methods: expected_methods || []
          } : null,
          metadata: {
            source: 'code_chat',
            error_category,
            expected_methods
          }
        })
      });

      if (feedbackResponse.ok) {
        const result = await feedbackResponse.json();
        res.json({
          success: true,
          feedback_id: result.feedback_id || response_id
        });
      } else {
        throw new Error('Failed to store feedback');
      }

    } catch (error) {
      monitoring.error('Code feedback submission failed', error);
      res.status(500).json({ error: error.message });
    }
  });

  /**
   * GET /api/code/stats
   * Get code assistance statistics
   */
  router.get('/stats', async (req, res) => {
    try {
      const { project, days = 7 } = req.query;

      // Get stats from Python service
      const statsResponse = await fetch(`${pythonServiceUrl}/roslyn/stats`);

      if (statsResponse.ok) {
        const stats = await statsResponse.json();
        res.json({
          code_entities: stats,
          timestamp: new Date().toISOString()
        });
      } else {
        throw new Error('Failed to fetch stats');
      }

    } catch (error) {
      monitoring.error('Code stats fetch failed', error);
      res.status(500).json({ error: error.message });
    }
  });

  return router;
}
