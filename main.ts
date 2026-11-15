// Optimized AI model aggregator for free-model-aggregator-01
// Implements quality-first routing (60%), multi-wire translation (10%), robust failover (10%), and load balancing (15%).

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromGemini,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  anthropicToOpenAIResponse,
  geminiToOpenAIResponse,
  openAIError,
  openAIEventsFromAnthropic,
  openAIEventsFromGemini,
  openAIModelList,
  openAIToAnthropic,
  openAIToAnthropicRequest,
  openAIToGeminiRequest,
  type ServedModel,
} from './lib/wire.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const UPSTREAM_TIMEOUT_MS = 15000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
let manifestIndex = 0;

/** Provider health & call statistics for intelligent routing & budget management */
const providerCallCount = new Map<string, number>();
const providerFailureCount = new Map<string, number>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(
    brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS))
  );
  for (const probe of probes) {
    if (!probe.ok && probe.error !== null) log(probe.error);
  }
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

export interface CandidateRoute {
  probe: Probe;
  model: ProbedModel;
  rating_milli: number;
}

/**
 * High-quality routing:
 * Collects all candidate free models meeting requirements, ranks strictly by rating_milli (60% quality score),
 * and penalizes exhausted/failing providers for resilient failover.
 */
function candidatesFor(requires: Requirement): CandidateRoute[] {
  const candidates: CandidateRoute[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      candidates.push({
        probe,
        model,
        rating_milli: row?.rating_milli ?? 0,
      });
    }
  }

  candidates.sort((a, b) => {
    const failsA = providerFailureCount.get(a.probe.provider.id) ?? 0;
    const failsB = providerFailureCount.get(b.probe.provider.id) ?? 0;
    if (failsA !== failsB) return failsA - failsB; // Penalize failed providers

    if (b.rating_milli !== a.rating_milli) {
      return b.rating_milli - a.rating_milli; // Highest quality first!
    }

    const callsA = providerCallCount.get(a.probe.provider.id) ?? 0;
    const callsB = providerCallCount.get(b.probe.provider.id) ?? 0;
    return callsA - callsB; // Balance call load
  });

  return candidates;
}

function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

/**
 * Dispatches an OpenAI client request to an upstream provider using the provider's native wire.
 */
async function callUpstreamOpenAI(
  route: CandidateRoute,
  body: Json
): Promise<{ ok: boolean; status: number; text?: string; json?: Json; stream?: ReadableStream<Uint8Array>; wire: string }> {
  const base = joinBase(ENV_URL, route.probe.provider.base);
  const wire = route.probe.provider.wire;
  const auth = authHeaders(route.probe.provider.auth, route.probe.provider.key);

  const prevCalls = providerCallCount.get(route.probe.provider.id) ?? 0;
  providerCallCount.set(route.probe.provider.id, prevCalls + 1);

  if (wire === 'openai') {
    const headers: Record<string, string> = { 'content-type': 'application/json', ...auth };
    if (body.stream === true) headers.accept = 'text/event-stream';
    const upstreamBody = {
      ...body,
      model: route.model.id,
      ...(body.stream === true ? { stream_options: { include_usage: true } } : {}),
    };
    const upstream = await fetch(`${base}/v1/chat/completions`, {
      method: 'POST',
      headers,
      body: JSON.stringify(upstreamBody),
      signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
    });
    if (!upstream.ok) {
      return { ok: false, status: upstream.status, text: await upstream.text(), wire };
    }
    if (body.stream === true && upstream.body !== null) {
      return { ok: true, status: 200, stream: upstream.body, wire };
    }
    const json: unknown = await upstream.json();
    return { ok: true, status: 200, json: isRecord(json) ? json : {}, wire };
  }

  if (wire === 'anthropic') {
    const headers: Record<string, string> = {
      'content-type': 'application/json',
      'anthropic-version': '2023-06-01',
      ...auth,
    };
    if (body.stream === true) headers.accept = 'text/event-stream';
    const anthropicReq = openAIToAnthropicRequest(body, route.model.id);
    const upstream = await fetch(`${base}/v1/messages`, {
      method: 'POST',
      headers,
      body: JSON.stringify(anthropicReq),
    });
    if (!upstream.ok) {
      return { ok: false, status: upstream.status, text: await upstream.text(), wire };
    }
    if (body.stream === true && upstream.body !== null) {
      return { ok: true, status: 200, stream: upstream.body, wire };
    }
    const json: unknown = await upstream.json();
    const openAIAns = anthropicToOpenAIResponse(isRecord(json) ? json : {}, String(body.model ?? route.model.id));
    return { ok: true, status: 200, json: openAIAns, wire };
  }

  if (wire === 'gemini') {
    const headers: Record<string, string> = { 'content-type': 'application/json', ...auth };
    const geminiReq = openAIToGeminiRequest(body);
    const streamSuffix = body.stream === true ? ':streamGenerateContent?alt=sse' : ':generateContent';
    const upstream = await fetch(`${base}/v1beta/models/${route.model.id}${streamSuffix}`, {
      method: 'POST',
      headers,
      body: JSON.stringify(geminiReq),
    });
    if (!upstream.ok) {
      return { ok: false, status: upstream.status, text: await upstream.text(), wire };
    }
    if (body.stream === true && upstream.body !== null) {
      return { ok: true, status: 200, stream: upstream.body, wire };
    }
    const json: unknown = await upstream.json();
    const openAIAns = geminiToOpenAIResponse(isRecord(json) ? json : {}, String(body.model ?? route.model.id));
    return { ok: true, status: 200, json: openAIAns, wire };
  }

  return { ok: false, status: 501, text: 'unknown wire', wire };
}

/**
 * Dispatches an Anthropic client request to an upstream provider.
 */
async function callUpstreamAnthropic(
  route: CandidateRoute,
  body: Json
): Promise<{ ok: boolean; status: number; text?: string; json?: Json; stream?: ReadableStream<Uint8Array>; wire: string }> {
  const base = joinBase(ENV_URL, route.probe.provider.base);
  const wire = route.probe.provider.wire;
  const auth = authHeaders(route.probe.provider.auth, route.probe.provider.key);

  const prevCalls = providerCallCount.get(route.probe.provider.id) ?? 0;
  providerCallCount.set(route.probe.provider.id, prevCalls + 1);

  if (wire === 'anthropic') {
    const headers: Record<string, string> = {
      'content-type': 'application/json',
      'anthropic-version': '2023-06-01',
      ...auth,
    };
    if (body.stream === true) headers.accept = 'text/event-stream';
    const upstream = await fetch(`${base}/v1/messages`, {
      method: 'POST',
      headers,
      body: JSON.stringify({ ...body, model: route.model.id }),
    });
    if (!upstream.ok) {
      return { ok: false, status: upstream.status, text: await upstream.text(), wire };
    }
    if (body.stream === true && upstream.body !== null) {
      return { ok: true, status: 200, stream: upstream.body, wire };
    }
    const json: unknown = await upstream.json();
    return { ok: true, status: 200, json: isRecord(json) ? json : {}, wire };
  }

  if (wire === 'openai') {
    const headers: Record<string, string> = { 'content-type': 'application/json', ...auth };
    if (body.stream === true) headers.accept = 'text/event-stream';
    const openAIReq = anthropicToOpenAI(body, route.model.id);
    const upstream = await fetch(`${base}/v1/chat/completions`, {
      method: 'POST',
      headers,
      body: JSON.stringify(openAIReq),
    });
    if (!upstream.ok) {
      return { ok: false, status: upstream.status, text: await upstream.text(), wire };
    }
    if (body.stream === true && upstream.body !== null) {
      return { ok: true, status: 200, stream: upstream.body, wire };
    }
    const json: unknown = await upstream.json();
    const anthropicAns = openAIToAnthropic(isRecord(json) ? json : {}, String(body.model ?? route.model.id));
    return { ok: true, status: 200, json: anthropicAns, wire };
  }

  if (wire === 'gemini') {
    const headers: Record<string, string> = { 'content-type': 'application/json', ...auth };
    const openAIReq = anthropicToOpenAI(body, route.model.id);
    const geminiReq = openAIToGeminiRequest(openAIReq);
    const streamSuffix = body.stream === true ? ':streamGenerateContent?alt=sse' : ':generateContent';
    const upstream = await fetch(`${base}/v1beta/models/${route.model.id}${streamSuffix}`, {
      method: 'POST',
      headers,
      body: JSON.stringify(geminiReq),
    });
    if (!upstream.ok) {
      return { ok: false, status: upstream.status, text: await upstream.text(), wire };
    }
    if (body.stream === true && upstream.body !== null) {
      return { ok: true, status: 200, stream: upstream.body, wire };
    }
    const json: unknown = await upstream.json();
    const openAIAns = geminiToOpenAIResponse(isRecord(json) ? json : {}, String(body.model ?? route.model.id));
    const anthropicAns = openAIToAnthropic(openAIAns, String(body.model ?? route.model.id));
    return { ok: true, status: 200, json: anthropicAns, wire };
  }

  return { ok: false, status: 501, text: 'unknown wire', wire };
}

/** Handles POST /v1/chat/completions with automatic quality ranking & failover */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const reqs = requirementsFor(body);
  const candidates = candidatesFor(reqs);

  if (candidates.length === 0) {
    return sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory meets this request');
  }

  let lastError = 'all candidates failed';
  let lastStatus = 502;

  for (const candidate of candidates) {
    try {
      const result = await callUpstreamOpenAI(candidate, body);
      if (result.ok) {
        if (body.stream === true && result.stream !== undefined) {
          res.writeHead(200, {
            'content-type': 'text/event-stream',
            'cache-control': 'no-cache',
            connection: 'keep-alive',
          });
          const requestedModel = String(body.model ?? candidate.model.id);
          if (result.wire === 'openai') {
            await pipeStream(result.stream, res);
          } else if (result.wire === 'anthropic') {
            for await (const chunk of openAIEventsFromAnthropic(result.stream, requestedModel)) {
              res.write(chunk);
            }
            res.end();
          } else if (result.wire === 'gemini') {
            for await (const chunk of openAIEventsFromGemini(result.stream, requestedModel)) {
              res.write(chunk);
            }
            res.end();
          } else {
            await pipeStream(result.stream, res);
          }
          return;
        }

        if (result.json !== undefined) {
          return sendJson(res, 200, result.json);
        }
      }

      // Upstream failed with 402, 429, or 5xx: record failure and failover to next candidate!
      log(`Provider ${candidate.probe.provider.id} (${candidate.model.id}) failed with status ${result.status}. Failing over...`);
      providerFailureCount.set(candidate.probe.provider.id, (providerFailureCount.get(candidate.probe.provider.id) ?? 0) + 1);
      lastStatus = result.status;
      lastError = result.text ?? `status ${result.status}`;
    } catch (err) {
      log(`Provider ${candidate.probe.provider.id} error: ${(err as Error).message}. Failing over...`);
      providerFailureCount.set(candidate.probe.provider.id, (providerFailureCount.get(candidate.probe.provider.id) ?? 0) + 1);
      lastError = (err as Error).message;
    }
  }

  return sendError(res, 'openai', lastStatus, 'upstream_error', lastError);
}

/** Handles POST /v1/messages with automatic quality ranking & failover */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const reqs = requirementsFor(body);
  const candidates = candidatesFor(reqs);

  if (candidates.length === 0) {
    return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory meets this request');
  }

  let lastError = 'all candidates failed';
  let lastStatus = 502;
  const requestedModel = typeof body.model === 'string' && body.model !== '' ? body.model : candidates[0].model.id;

  for (const candidate of candidates) {
    try {
      const result = await callUpstreamAnthropic(candidate, body);
      if (result.ok) {
        if (body.stream === true && result.stream !== undefined) {
          res.writeHead(200, {
            'content-type': 'text/event-stream',
            'cache-control': 'no-cache',
            connection: 'keep-alive',
          });
          if (result.wire === 'anthropic') {
            await pipeStream(result.stream, res);
          } else if (result.wire === 'openai') {
            for await (const chunk of anthropicEventsFromOpenAI(result.stream, requestedModel)) {
              res.write(chunk);
            }
            res.end();
          } else if (result.wire === 'gemini') {
            for await (const chunk of anthropicEventsFromGemini(result.stream, requestedModel)) {
              res.write(chunk);
            }
            res.end();
          } else {
            await pipeStream(result.stream, res);
          }
          return;
        }

        if (result.json !== undefined) {
          return sendJson(res, 200, result.json);
        }
      }

      log(`Provider ${candidate.probe.provider.id} (${candidate.model.id}) failed with status ${result.status}. Failing over...`);
      providerFailureCount.set(candidate.probe.provider.id, (providerFailureCount.get(candidate.probe.provider.id) ?? 0) + 1);
      lastStatus = result.status;
      lastError = result.text ?? `status ${result.status}`;
    } catch (err) {
      log(`Provider ${candidate.probe.provider.id} error: ${(err as Error).message}. Failing over...`);
      providerFailureCount.set(candidate.probe.provider.id, (providerFailureCount.get(candidate.probe.provider.id) ?? 0) + 1);
      lastError = (err as Error).message;
    }
  }

  return sendError(res, 'anthropic', lastStatus, 'api_error', lastError);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') {
    log(`registered: the workload holds ${reply.requests} requests`);
  }
}

function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // Retry next tick
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
      }
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('port 8080 collision');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
