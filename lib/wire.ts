// Enhanced wire converters supporting OpenAI, Anthropic, and Gemini wires bidirectionally.
import { isRecord, type Json } from './providers.ts';

export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

const FIXED_OPENAI_CREATED = 1767225600;
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {}
  }
  return {};
}

export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** Anthropic Messages -> OpenAI Chat Completions */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** OpenAI Chat Completions answer -> Anthropic Messages object */
export function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

/** OpenAI Chat Completions -> Anthropic Messages Request */
export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const messages: Json[] = [];
  let system: string | undefined = undefined;
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      system = typeof raw.content === 'string' ? raw.content : textOf(raw.content);
      continue;
    }
    if (raw.role === 'user') {
      if (typeof raw.content === 'string') {
        messages.push({ role: 'user', content: raw.content });
      } else if (Array.isArray(raw.content)) {
        const blocks: Json[] = [];
        for (const part of raw.content) {
          if (!isRecord(part)) continue;
          if (part.type === 'text' && typeof part.text === 'string') {
            blocks.push({ type: 'text', text: part.text });
          } else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
            const url = part.image_url.url;
            if (url.startsWith('data:')) {
              const comma = url.indexOf(',');
              const meta = url.slice(5, comma);
              const media_type = meta.split(';')[0] || 'image/png';
              const data = url.slice(comma + 1);
              blocks.push({ type: 'image', source: { type: 'base64', media_type, data } });
            } else {
              blocks.push({ type: 'image', source: { type: 'url', url } });
            }
          }
        }
        messages.push({ role: 'user', content: blocks.length > 0 ? blocks : '' });
      }
    } else if (raw.role === 'assistant') {
      const blocks: Json[] = [];
      if (typeof raw.content === 'string' && raw.content !== '') {
        blocks.push({ type: 'text', text: raw.content });
      }
      if (Array.isArray(raw.tool_calls)) {
        for (const tc of raw.tool_calls) {
          if (isRecord(tc) && isRecord(tc.function)) {
            blocks.push({
              type: 'tool_use',
              id: typeof tc.id === 'string' ? tc.id : 'toolu_unknown',
              name: typeof tc.function.name === 'string' ? tc.function.name : '',
              input: argumentsOf(tc.function.arguments),
            });
          }
        }
      }
      messages.push({ role: 'assistant', content: blocks.length > 0 ? blocks : '' });
    } else if (raw.role === 'tool') {
      messages.push({
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
            content: typeof raw.content === 'string' ? raw.content : JSON.stringify(raw.content ?? ''),
          },
        ],
      });
    }
  }
  const out: Json = {
    model,
    messages,
    max_tokens: typeof body.max_tokens === 'number' ? body.max_tokens : 4096,
  };
  if (system !== undefined && system !== '') out.system = system;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((t) => {
      if (!isRecord(t) || !isRecord(t.function) || typeof t.function.name !== 'string') return [];
      return [
        {
          name: t.function.name,
          description: typeof t.function.description === 'string' ? t.function.description : '',
          input_schema: isRecord(t.function.parameters) ? t.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** Anthropic Messages answer -> OpenAI Chat Completions response */
export function anthropicToOpenAIResponse(body: Json, model: string): Json {
  const contentBlocks = Array.isArray(body.content) ? body.content : [];
  let text = '';
  const toolCalls: Json[] = [];
  for (const b of contentBlocks) {
    if (!isRecord(b)) continue;
    if (b.type === 'text' && typeof b.text === 'string') text += b.text;
    if (b.type === 'tool_use') {
      toolCalls.push({
        id: typeof b.id === 'string' ? b.id : 'call_unknown',
        type: 'function',
        function: {
          name: typeof b.name === 'string' ? b.name : '',
          arguments: JSON.stringify(isRecord(b.input) ? b.input : {}),
        },
      });
    }
  }
  let finish_reason = 'stop';
  if (body.stop_reason === 'max_tokens') finish_reason = 'length';
  else if (body.stop_reason === 'tool_use') finish_reason = 'tool_calls';
  else if (body.stop_reason === 'refusal') finish_reason = 'content_filter';
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: typeof body.id === 'string' ? body.id : 'chatcmpl_aggregator',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: {
          role: 'assistant',
          content: text !== '' ? text : null,
          ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
        },
        finish_reason,
      },
    ],
    usage: {
      prompt_tokens: numberOrZero(usage.input_tokens),
      completion_tokens: numberOrZero(usage.output_tokens),
      total_tokens: numberOrZero(usage.input_tokens) + numberOrZero(usage.output_tokens),
    },
  };
}

/** OpenAI -> Gemini generateContent Request */
export function openAIToGeminiRequest(body: Json): Json {
  const contents: Json[] = [];
  let systemInstruction: Json | undefined = undefined;
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = typeof raw.content === 'string' ? raw.content : textOf(raw.content);
      if (text !== '') systemInstruction = { parts: [{ text }] };
      continue;
    }
    const role = raw.role === 'assistant' ? 'model' : 'user';
    const parts: Json[] = [];
    if (typeof raw.content === 'string' && raw.content !== '') {
      parts.push({ text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const p of raw.content) {
        if (!isRecord(p)) continue;
        if (p.type === 'text' && typeof p.text === 'string') parts.push({ text: p.text });
        else if (p.type === 'image_url' && isRecord(p.image_url) && typeof p.image_url.url === 'string') {
          const url = p.image_url.url;
          if (url.startsWith('data:')) {
            const comma = url.indexOf(',');
            const meta = url.slice(5, comma);
            const mimeType = meta.split(';')[0] || 'image/png';
            const data = url.slice(comma + 1);
            parts.push({ inlineData: { mimeType, data } });
          }
        }
      }
    }
    if (parts.length > 0) contents.push({ role, parts });
  }

  const out: Json = { contents };
  if (systemInstruction !== undefined) out.systemInstruction = systemInstruction;
  const genConfig: Json = {};
  if (typeof body.temperature === 'number') genConfig.temperature = body.temperature;
  if (typeof body.top_p === 'number') genConfig.topP = body.top_p;
  if (typeof body.max_tokens === 'number') genConfig.maxOutputTokens = body.max_tokens;
  if (Array.isArray(body.stop) && body.stop.length > 0) {
    genConfig.stopSequences = body.stop.filter((s): s is string => typeof s === 'string');
  } else if (typeof body.stop === 'string' && body.stop !== '') {
    genConfig.stopSequences = [body.stop];
  }
  if (isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema')) {
    genConfig.responseMimeType = 'application/json';
  }
  if (Object.keys(genConfig).length > 0) out.generationConfig = genConfig;

  if (Array.isArray(body.tools) && body.tools.length > 0) {
    const declarations = body.tools.flatMap((t) => {
      if (!isRecord(t) || !isRecord(t.function) || typeof t.function.name !== 'string') return [];
      return [
        {
          name: t.function.name,
          description: typeof t.function.description === 'string' ? t.function.description : '',
          parameters: isRecord(t.function.parameters) ? t.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  return out;
}

/** Gemini response -> OpenAI Chat Completions */
export function geminiToOpenAIResponse(body: Json, model: string): Json {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  let text = '';
  const toolCalls: Json[] = [];
  for (let i = 0; i < parts.length; i++) {
    const p = parts[i];
    if (!isRecord(p)) continue;
    if (typeof p.text === 'string') text += p.text;
    if (isRecord(p.functionCall) && typeof p.functionCall.name === 'string') {
      toolCalls.push({
        id: `call_${i}`,
        type: 'function',
        function: {
          name: p.functionCall.name,
          arguments: JSON.stringify(isRecord(p.functionCall.args) ? p.functionCall.args : {}),
        },
      });
    }
  }
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    id: 'chatcmpl_gemini',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: {
          role: 'assistant',
          content: text !== '' ? text : null,
          ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
        },
        finish_reason: candidate.finishReason === 'MAX_TOKENS' ? 'length' : 'stop',
      },
    ],
    usage: {
      prompt_tokens: numberOrZero(usage.promptTokenCount),
      completion_tokens: numberOrZero(usage.candidatesTokenCount),
      total_tokens: numberOrZero(usage.totalTokenCount),
    },
  };
}

/** OpenAI SSE from Anthropic SSE Stream */
export async function* openAIEventsFromAnthropic(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let callId = '';
  let callName = '';
  let toolIndex = 0;
  let inputTokens = 0;
  let outputTokens = 0;
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const line of frame.split('\n')) {
          const trimmed = line.trim();
          if (!trimmed.startsWith('data:')) continue;
          const payload = trimmed.slice(5).trim();
          let eventObj: unknown;
          try {
            eventObj = JSON.parse(payload);
          } catch {
            continue;
          }
          if (!isRecord(eventObj)) continue;
          if (eventObj.type === 'message_start' && isRecord(eventObj.message) && isRecord(eventObj.message.usage)) {
            inputTokens = numberOrZero(eventObj.message.usage.input_tokens);
          } else if (eventObj.type === 'content_block_start' && isRecord(eventObj.content_block)) {
            if (eventObj.content_block.type === 'tool_use') {
              callId = typeof eventObj.content_block.id === 'string' ? eventObj.content_block.id : `call_${toolIndex}`;
              callName = typeof eventObj.content_block.name === 'string' ? eventObj.content_block.name : '';
              yield sseEvent(null, {
                id: 'chatcmpl_stream',
                object: 'chat.completion.chunk',
                created: FIXED_OPENAI_CREATED,
                model,
                choices: [
                  {
                    index: 0,
                    delta: {
                      tool_calls: [
                        {
                          index: toolIndex,
                          id: callId,
                          type: 'function',
                          function: { name: callName, arguments: '' },
                        },
                      ],
                    },
                    finish_reason: null,
                  },
                ],
              });
            }
          } else if (eventObj.type === 'content_block_delta' && isRecord(eventObj.delta)) {
            if (eventObj.delta.type === 'text_delta' && typeof eventObj.delta.text === 'string') {
              yield sseEvent(null, {
                id: 'chatcmpl_stream',
                object: 'chat.completion.chunk',
                created: FIXED_OPENAI_CREATED,
                model,
                choices: [{ index: 0, delta: { content: eventObj.delta.text }, finish_reason: null }],
              });
            } else if (eventObj.delta.type === 'input_json_delta' && typeof eventObj.delta.partial_json === 'string') {
              yield sseEvent(null, {
                id: 'chatcmpl_stream',
                object: 'chat.completion.chunk',
                created: FIXED_OPENAI_CREATED,
                model,
                choices: [
                  {
                    index: 0,
                    delta: {
                      tool_calls: [
                        {
                          index: toolIndex,
                          function: { arguments: eventObj.delta.partial_json },
                        },
                      ],
                    },
                    finish_reason: null,
                  },
                ],
              });
            }
          } else if (eventObj.type === 'content_block_stop') {
            toolIndex += 1;
          } else if (eventObj.type === 'message_delta' && isRecord(eventObj.delta)) {
            if (isRecord(eventObj.usage)) {
              outputTokens = numberOrZero(eventObj.usage.output_tokens);
            }
            const fr = eventObj.delta.stop_reason === 'tool_use' ? 'tool_calls' : eventObj.delta.stop_reason === 'max_tokens' ? 'length' : 'stop';
            yield sseEvent(null, {
              id: 'chatcmpl_stream',
              object: 'chat.completion.chunk',
              created: FIXED_OPENAI_CREATED,
              model,
              choices: [{ index: 0, delta: {}, finish_reason: fr }],
            });
          } else if (eventObj.type === 'message_stop') {
            yield sseEvent(null, {
              id: 'chatcmpl_stream',
              object: 'chat.completion.chunk',
              created: FIXED_OPENAI_CREATED,
              model,
              choices: [],
              usage: {
                prompt_tokens: inputTokens,
                completion_tokens: outputTokens,
                total_tokens: inputTokens + outputTokens,
              },
            });
            yield 'data: [DONE]\n\n';
          }
        }
        at = buffer.indexOf('\n\n');
      }
    }
  } finally {
    reader.releaseLock();
  }
}

/** Anthropic SSE events from OpenAI Chat Completions stream */
export async function* anthropicEventsFromOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
    ];
  };
  const closeEvents = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    toolIndexes.clear();
    return out;
  };
  const finishEvents = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    return [
      ...closeEvents(),
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') {
        out.push(...startEvents(), ...finishEvents(pendingStop ?? 'end_turn'));
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        inputTokens = numberOrZero(chunk.usage.prompt_tokens);
        outputTokens = numberOrZero(chunk.usage.completion_tokens);
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      out.push(...startEvents());
      if (typeof delta.content === 'string' && delta.content !== '') {
        if (textIndex === -1) {
          textIndex = nextIndex;
          nextIndex += 1;
          out.push(sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }));
        }
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } }));
      }
      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call)) continue;
        const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
        let index = toolIndexes.get(upstreamIndex);
        if (index === undefined) {
          if (textIndex !== -1) {
            out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
            textIndex = -1;
          }
          index = nextIndex;
          nextIndex += 1;
          toolIndexes.set(upstreamIndex, index);
          const fn = isRecord(call.function) ? call.function : {};
          out.push(
            sseEvent('content_block_start', {
              type: 'content_block_start',
              index,
              content_block: {
                type: 'tool_use',
                id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
                name: typeof fn.name === 'string' ? fn.name : '',
                input: {},
              },
            }),
          );
        }
        const fn = isRecord(call.function) ? call.function : {};
        if (typeof fn.arguments === 'string' && fn.arguments !== '') {
          out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
        }
      }
      if (choice.finish_reason !== null && choice.finish_reason !== undefined) pendingStop = stopReasonOf(choice.finish_reason);
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const event of eventsOfFrame(frame)) yield event;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const event of eventsOfFrame(buffer)) yield event;
  } finally {
    reader.releaseLock();
  }
  if (!finished) {
    for (const event of startEvents()) yield event;
    for (const event of finishEvents(pendingStop ?? 'end_turn')) yield event;
  }
}

/** OpenAI SSE events from Gemini SSE Stream */
export async function* openAIEventsFromGemini(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let sawFinish = false;
  let promptTokens = 0;
  let completionTokens = 0;

  yield sseEvent(null, {
    id: 'chatcmpl_gemini_stream',
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }],
  });

  const processLine = (line: string): string[] => {
    const trimmed = line.trim();
    if (!trimmed.startsWith('data:')) return [];
    const payload = trimmed.slice(5).trim();
    if (payload === '' || payload === '[DONE]') return [];
    let obj: unknown;
    try {
      obj = JSON.parse(payload);
    } catch {
      return [];
    }
    if (!isRecord(obj)) return [];
    const out: string[] = [];

    if (isRecord(obj.usageMetadata)) {
      const u = obj.usageMetadata as Record<string, unknown>;
      if (typeof u.promptTokenCount === 'number') promptTokens = u.promptTokenCount;
      if (typeof u.candidatesTokenCount === 'number') completionTokens = u.candidatesTokenCount;
    }

    const candidates = Array.isArray(obj.candidates) ? obj.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate !== null) {
      const content = isRecord(candidate.content) ? candidate.content : {};
      const parts = Array.isArray(content.parts) ? content.parts : [];
      for (const p of parts) {
        if (isRecord(p) && typeof p.text === 'string' && p.text !== '') {
          out.push(
            sseEvent(null, {
              id: 'chatcmpl_gemini_stream',
              object: 'chat.completion.chunk',
              created: FIXED_OPENAI_CREATED,
              model,
              choices: [{ index: 0, delta: { content: p.text }, finish_reason: null }],
            })
          );
        }
      }
      if (typeof candidate.finishReason === 'string' && !sawFinish) {
        sawFinish = true;
        const fr = candidate.finishReason === 'MAX_TOKENS' ? 'length' : 'stop';
        out.push(
          sseEvent(null, {
            id: 'chatcmpl_gemini_stream',
            object: 'chat.completion.chunk',
            created: FIXED_OPENAI_CREATED,
            model,
            choices: [{ index: 0, delta: {}, finish_reason: fr }],
          })
        );
      }
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const line of frame.split('\n')) {
          for (const ev of processLine(line)) yield ev;
        }
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') {
      for (const line of buffer.split('\n')) {
        for (const ev of processLine(line)) yield ev;
      }
    }
  } finally {
    reader.releaseLock();
  }

  if (!sawFinish) {
    yield sseEvent(null, {
      id: 'chatcmpl_gemini_stream',
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [{ index: 0, delta: {}, finish_reason: 'stop' }],
    });
  }

  yield sseEvent(null, {
    id: 'chatcmpl_gemini_stream',
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [],
    usage: {
      prompt_tokens: promptTokens,
      completion_tokens: completionTokens,
      total_tokens: promptTokens + completionTokens,
    },
  });

  yield 'data: [DONE]\n\n';
}

/** Anthropic SSE events from Gemini SSE Stream */
export async function* anthropicEventsFromGemini(
  upstream: ReadableStream<Uint8Array>,
  model: string,
  estimatedInputTokens: number = 0
): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let outputTokens = 0;
  let fullText = '';

  yield sseEvent('message_start', {
    type: 'message_start',
    message: {
      id: 'msg_gemini_stream',
      type: 'message',
      role: 'assistant',
      model,
      content: [],
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: estimatedInputTokens, output_tokens: 0 },
    },
  });

  yield sseEvent('content_block_start', {
    type: 'content_block_start',
    index: 0,
    content_block: { type: 'text', text: '' },
  });

  const processLine = (line: string): string[] => {
    const trimmed = line.trim();
    if (!trimmed.startsWith('data:')) return [];
    const payload = trimmed.slice(5).trim();
    if (payload === '' || payload === '[DONE]') return [];
    let obj: unknown;
    try {
      obj = JSON.parse(payload);
    } catch {
      return [];
    }
    if (!isRecord(obj)) return [];
    const out: string[] = [];

    if (isRecord(obj.usageMetadata)) {
      const u = obj.usageMetadata as Record<string, unknown>;
      if (typeof u.candidatesTokenCount === 'number') outputTokens = u.candidatesTokenCount;
    }

    const candidates = Array.isArray(obj.candidates) ? obj.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate !== null) {
      const content = isRecord(candidate.content) ? candidate.content : {};
      const parts = Array.isArray(content.parts) ? content.parts : [];
      for (const p of parts) {
        if (isRecord(p) && typeof p.text === 'string' && p.text !== '') {
          fullText += p.text;
          out.push(
            sseEvent('content_block_delta', {
              type: 'content_block_delta',
              index: 0,
              delta: { type: 'text_delta', text: p.text },
            })
          );
        }
      }
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const line of frame.split('\n')) {
          for (const ev of processLine(line)) yield ev;
        }
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') {
      for (const line of buffer.split('\n')) {
        for (const ev of processLine(line)) yield ev;
      }
    }
  } finally {
    reader.releaseLock();
  }

  if (outputTokens === 0) {
    outputTokens = Math.max(1, Math.floor(fullText.length / 4));
  }

  yield sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 });
  yield sseEvent('message_delta', {
    type: 'message_delta',
    delta: { stop_reason: 'end_turn', stop_sequence: null },
    usage: { output_tokens: outputTokens },
  });
  yield sseEvent('message_stop', { type: 'message_stop' });
}
