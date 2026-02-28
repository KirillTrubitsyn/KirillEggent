import {
  streamText,
  generateText,
  stepCountIs,
  type ModelMessage,
  type UserContent,
  type ToolExecutionOptions,
  type ToolSet,
} from "ai";
import { createModel } from "@/lib/providers/llm-provider";
import { buildSystemPrompt } from "@/lib/agent/prompts";
import { getSettings } from "@/lib/storage/settings-store";
import { getChat, saveChat } from "@/lib/storage/chat-store";
import { createAgentTools } from "@/lib/tools/tool";
import { getProjectMcpTools } from "@/lib/mcp/client";
import type { AgentContext } from "@/lib/agent/types";
import type { Attachment, ChatMessage, Chat, AppSettings } from "@/lib/types";
import { publishUiSyncEvent } from "@/lib/realtime/event-bus";
import fs from "fs/promises";

const LLM_LOG_BORDER = "═".repeat(60);

/**
 * Maximum number of recent messages to keep in history when calling the LLM.
 * This prevents the prompt from growing unboundedly as the conversation continues.
 * Tool-call / tool-result pairs are counted individually.
 */
const MAX_HISTORY_MESSAGES = 40;

/**
 * Trim history to the most recent messages while keeping tool-call/tool-result
 * pairs intact (never orphan a tool result without its preceding call).
 */
function trimHistory(messages: ModelMessage[]): ModelMessage[] {
  if (messages.length <= MAX_HISTORY_MESSAGES) {
    return messages;
  }

  // Take the tail of the history
  let startIndex = messages.length - MAX_HISTORY_MESSAGES;

  // Ensure we don't start on a tool-result message (orphaned from its call).
  // Walk forward until we find a user or assistant message.
  while (startIndex < messages.length && messages[startIndex].role === "tool") {
    startIndex++;
  }

  const trimmed = messages.slice(startIndex);
  if (trimmed.length < messages.length) {
    console.log(
      `[Agent] Trimmed history from ${messages.length} to ${trimmed.length} messages`
    );
  }
  return trimmed;
}

// ---------------------------------------------------------------------------
// Context optimisation: tool-result truncation + history summarisation
// ---------------------------------------------------------------------------

/**
 * Maximum character length for tool results in older messages.
 * Recent messages (last TOOL_RESULT_RECENT_COUNT) keep full results.
 */
const MAX_OLD_TOOL_RESULT_CHARS = 500;

/**
 * Number of most-recent messages whose tool results are kept intact.
 */
const TOOL_RESULT_RECENT_COUNT = 10;

/**
 * Number of most-recent messages kept verbatim when summarisation is used.
 */
const RECENT_MESSAGES_WINDOW = 20;

/**
 * Truncate tool-result outputs (and large tool-call inputs) in older messages
 * so they don't bloat the context window.
 */
function truncateOldToolResults(messages: ModelMessage[]): ModelMessage[] {
  const recentStart = Math.max(0, messages.length - TOOL_RESULT_RECENT_COUNT);

  return messages.map((msg, index) => {
    if (index >= recentStart) return msg;

    // --- Truncate tool result outputs ---
    if (msg.role === "tool" && Array.isArray(msg.content)) {
      const newContent = (msg.content as Array<Record<string, unknown>>).map(
        (part) => {
          if (part.type !== "tool-result") return part;

          const output = part.output;
          let outputStr: string;
          if (
            output &&
            typeof output === "object" &&
            "value" in (output as Record<string, unknown>)
          ) {
            const val = (output as Record<string, unknown>).value;
            outputStr = typeof val === "string" ? val : JSON.stringify(val);
          } else {
            outputStr =
              typeof output === "string" ? output : JSON.stringify(output);
          }

          if (outputStr.length <= MAX_OLD_TOOL_RESULT_CHARS) return part;

          const toolName = (part.toolName as string) || "tool";
          return {
            ...part,
            output: {
              type: "json",
              value: `[${toolName}: ${outputStr.length} chars → ${MAX_OLD_TOOL_RESULT_CHARS}]\n${outputStr.slice(0, MAX_OLD_TOOL_RESULT_CHARS)}…`,
            },
          };
        },
      );
      return { ...msg, content: newContent } as ModelMessage;
    }

    // --- Truncate large tool-call inputs in assistant messages ---
    if (msg.role === "assistant" && Array.isArray(msg.content)) {
      const newContent = (msg.content as Array<Record<string, unknown>>).map(
        (part) => {
          if (part.type !== "tool-call") return part;

          const inputStr = JSON.stringify(part.input);
          if (inputStr.length <= MAX_OLD_TOOL_RESULT_CHARS) return part;

          const toolName = (part.toolName as string) || "tool";
          return {
            ...part,
            input: {
              _truncated: true,
              _summary: `[${toolName} input: ${inputStr.length} chars truncated]`,
            },
          };
        },
      );
      return { ...msg, content: newContent } as ModelMessage;
    }

    return msg;
  });
}

/**
 * Convert a ModelMessage to a compact text line for the summariser prompt.
 * Tool-result messages are skipped (too verbose, low summary value).
 */
function modelMessageToSummaryLine(msg: ModelMessage): string | null {
  if (msg.role === "user") {
    const content =
      typeof msg.content === "string"
        ? msg.content
        : JSON.stringify(msg.content);
    return `User: ${content.slice(0, 300)}`;
  }

  if (msg.role === "assistant") {
    if (typeof msg.content === "string") {
      return msg.content ? `Assistant: ${msg.content.slice(0, 300)}` : null;
    }
    if (Array.isArray(msg.content)) {
      const parts: string[] = [];
      for (const part of msg.content) {
        const p = part as Record<string, unknown>;
        if (p.type === "text" && typeof p.text === "string") {
          const t = (p.text as string).trim();
          if (t) parts.push(t.slice(0, 200));
        } else if (p.type === "tool-call" && typeof p.toolName === "string") {
          parts.push(`[called ${p.toolName}]`);
        }
      }
      return parts.length ? `Assistant: ${parts.join(" ")}` : null;
    }
    return null;
  }

  return null; // skip tool results
}

/**
 * Generate a concise summary of older messages using the utility model.
 */
async function generateHistorySummary(
  messages: ModelMessage[],
  settings: AppSettings,
): Promise<string> {
  const model = createModel(settings.utilityModel);

  const lines: string[] = [];
  for (const msg of messages) {
    const line = modelMessageToSummaryLine(msg);
    if (line) lines.push(line);
  }

  let conversationText = lines.join("\n");
  // Cap input to the summariser to avoid excessive cost
  if (conversationText.length > 12000) {
    conversationText =
      conversationText.slice(0, 12000) + "\n…[rest truncated]";
  }

  const result = await generateText({
    model,
    messages: [
      {
        role: "user",
        content:
          `Summarize this conversation between a user and an AI assistant.\n` +
          `Preserve: key topics discussed, decisions made, important facts, ` +
          `current task context, and any unresolved questions.\n` +
          `Be concise (max 400 words). Write in the same language as the conversation.\n\n` +
          conversationText,
      },
    ],
    temperature: 0.3,
    maxOutputTokens: 1024,
  });

  return result.text;
}

/**
 * Prepare an optimised message history for the LLM:
 *  1. Truncate tool results in older messages
 *  2. When history exceeds MAX_HISTORY_MESSAGES, summarise older messages
 *     via the utility model and keep only recent ones verbatim
 *
 * The summary is cached on the Chat object to avoid re-generation.
 */
async function prepareOptimizedHistory(
  allMessages: ModelMessage[],
  chat: Chat | null,
  settings: AppSettings,
): Promise<ModelMessage[]> {
  // Step 1: Always truncate old tool results
  const processed = truncateOldToolResults(allMessages);

  // Step 2: If within budget, return as-is
  if (processed.length <= MAX_HISTORY_MESSAGES) {
    return processed;
  }

  // Step 3: Split into old (to summarise) and recent (to keep verbatim)
  let recentStart = Math.max(0, processed.length - RECENT_MESSAGES_WINDOW);
  // Never orphan a tool result without its preceding call
  while (
    recentStart < processed.length &&
    processed[recentStart].role === "tool"
  ) {
    recentStart++;
  }

  const oldMessages = processed.slice(0, recentStart);
  const recentMessages = processed.slice(recentStart);

  if (oldMessages.length === 0) {
    return recentMessages;
  }

  // Step 4: Check whether a cached summary is still fresh enough
  const cacheValid =
    chat?.historySummary &&
    typeof chat.historySummaryUpToIndex === "number" &&
    Math.abs(chat.historySummaryUpToIndex - oldMessages.length) <= 5;

  let summary: string;

  if (cacheValid) {
    summary = chat!.historySummary!;
    console.log(
      `[Agent] Using cached history summary (covers ${chat!.historySummaryUpToIndex} messages)`,
    );
  } else {
    console.log(
      `[Agent] Generating history summary for ${oldMessages.length} old messages…`,
    );
    try {
      summary = await generateHistorySummary(oldMessages, settings);
      // Cache the summary on the chat object
      if (chat) {
        chat.historySummary = summary;
        chat.historySummaryUpToIndex = oldMessages.length;
        await saveChat(chat);
      }
      console.log(`[Agent] Summary generated (${summary.length} chars)`);
    } catch (err) {
      console.error(
        "[Agent] Summary generation failed, falling back to trimHistory:",
        err,
      );
      return trimHistory(processed);
    }
  }

  // Step 5: Build optimised history — summary pair + recent messages
  const optimized: ModelMessage[] = [
    {
      role: "user",
      content: `[Summary of earlier conversation (${oldMessages.length} messages)]:\n${summary}`,
    },
    {
      role: "assistant",
      content:
        "I have the context from our previous conversation. Let's continue.",
    },
    ...recentMessages,
  ];

  console.log(
    `[Agent] Optimized history: ${allMessages.length} → ${optimized.length} messages ` +
      `(summary + ${recentMessages.length} recent)`,
  );

  return optimized;
}

/**
 * After generateText completes, extract the final text.  When `generated.text`
 * is empty (e.g. generation stopped on a tool-call step) we scan all response
 * messages for the last non-empty assistant text so the user still gets an
 * answer rather than "Пустой ответ от агента".
 */
function extractFinalText(
  generated: { text: string; response?: { messages?: ModelMessage[] } }
): string {
  const primary = (generated.text ?? "").trim();
  if (primary) return primary;

  const responseMessages = generated.response?.messages;
  if (!Array.isArray(responseMessages) || responseMessages.length === 0) {
    return "";
  }

  // Walk response messages backwards looking for any assistant text
  for (let i = responseMessages.length - 1; i >= 0; i--) {
    const msg = responseMessages[i];
    if (msg.role !== "assistant") continue;

    const content = msg.content;
    if (typeof content === "string" && content.trim()) {
      return content.trim();
    }
    if (Array.isArray(content)) {
      for (const part of content) {
        if (
          typeof part === "object" &&
          part !== null &&
          "type" in part &&
          part.type === "text" &&
          "text" in part
        ) {
          const text = (part as { text: string }).text.trim();
          if (text) return text;
        }
      }
    }
  }

  return "";
}

function asRecord(value: unknown): Record<string, unknown> | null {
  if (value == null || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }
  return value as Record<string, unknown>;
}

function toStableValue(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map((item) => toStableValue(item));
  }
  const record = asRecord(value);
  if (!record) {
    return value;
  }
  return Object.keys(record)
    .sort()
    .reduce<Record<string, unknown>>((acc, key) => {
      acc[key] = toStableValue(record[key]);
      return acc;
    }, {});
}

function stableSerialize(value: unknown): string {
  try {
    return JSON.stringify(toStableValue(value));
  } catch {
    return String(value);
  }
}

function parseJsonObject(text: string): Record<string, unknown> | null {
  const trimmed = text.trim();
  if (!trimmed.startsWith("{") || !trimmed.endsWith("}")) {
    return null;
  }
  try {
    const parsed = JSON.parse(trimmed);
    return asRecord(parsed);
  } catch {
    return null;
  }
}

function extractDeterministicFailureSignature(output: unknown): string | null {
  const outputRecord = asRecord(output);
  if (outputRecord && outputRecord.success === false) {
    const errorText =
      typeof outputRecord.error === "string"
        ? outputRecord.error
        : "Tool returned success=false";
    const codeText = typeof outputRecord.code === "string" ? outputRecord.code : "";
    return [errorText, codeText].filter(Boolean).join(" | ");
  }

  if (typeof output !== "string") {
    return null;
  }

  const trimmed = output.trim();
  const parsed = parseJsonObject(trimmed);
  if (parsed && parsed.success === false) {
    const errorText =
      typeof parsed.error === "string" ? parsed.error : "Tool returned success=false";
    const codeText = typeof parsed.code === "string" ? parsed.code : "";
    return [errorText, codeText].filter(Boolean).join(" | ");
  }

  const isExplicitFailure =
    trimmed.startsWith("[MCP tool error]") ||
    trimmed.startsWith("[Preflight error]") ||
    trimmed.startsWith("[Loop guard]") ||
    /^Failed\b/i.test(trimmed) ||
    /^Skill ".+" not found\./i.test(trimmed) ||
    (/\bnot found\b/i.test(trimmed) &&
      !/No relevant memories found\./i.test(trimmed));

  if (!isExplicitFailure) {
    return null;
  }

  return trimmed.length > 400 ? `${trimmed.slice(0, 400)}...` : trimmed;
}

function applyGlobalToolLoopGuard(tools: ToolSet): ToolSet {
  const deterministicFailureByCall = new Map<string, string>();
  const wrappedTools: ToolSet = {};

  for (const [toolName, toolDef] of Object.entries(tools)) {
    if (typeof toolDef.execute !== "function") {
      wrappedTools[toolName] = toolDef;
      continue;
    }

    wrappedTools[toolName] = {
      ...toolDef,
      execute: async (input: unknown, options: ToolExecutionOptions) => {
        const callKey = `${toolName}:${stableSerialize(input)}`;
        const previousFailure = deterministicFailureByCall.get(callKey);
        if (previousFailure) {
          return (
            `[Loop guard] Blocked repeated tool call "${toolName}" with identical arguments.\n` +
            `Previous deterministic error: ${previousFailure}\n` +
            "Change arguments based on the tool error before retrying."
          );
        }

        const output = await toolDef.execute(input as never, options as never);
        const failureSignature = extractDeterministicFailureSignature(output);
        if (failureSignature) {
          deterministicFailureByCall.set(callKey, failureSignature);
        } else {
          deterministicFailureByCall.delete(callKey);
        }
        return output;
      },
    } as typeof toolDef;
  }

  return wrappedTools;
}

/**
 * Convert stored ChatMessages to AI SDK ModelMessage format
 */
function convertChatMessagesToModelMessages(messages: ChatMessage[]): ModelMessage[] {
  const result: ModelMessage[] = [];

  for (const m of messages) {
    if (m.role === "tool") {
      // Tool result message - AI SDK uses 'output' not 'result'
      result.push({
        role: "tool",
        content: [{
          type: "tool-result",
          toolCallId: m.toolCallId!,
          toolName: m.toolName!,
          output: { type: "json", value: m.toolResult as import("@ai-sdk/provider").JSONValue },
        }],
      });
    } else if (m.role === "assistant" && m.toolCalls && m.toolCalls.length > 0) {
      // Assistant message with tool calls - AI SDK uses 'input' not 'args'
      const content: Array<
        | { type: "text"; text: string }
        | { type: "tool-call"; toolCallId: string; toolName: string; input: unknown }
      > = [];
      if (m.content) {
        content.push({ type: "text", text: m.content });
      }
      for (const tc of m.toolCalls) {
        content.push({
          type: "tool-call",
          toolCallId: tc.toolCallId,
          toolName: tc.toolName,
          input: tc.args,
        });
      }
      result.push({ role: "assistant", content });
    } else if (m.role === "user" || m.role === "assistant") {
      // Regular user or assistant message
      result.push({ role: m.role, content: m.content });
    }
    // Skip system messages for now
  }

  return result;
}

/**
 * Convert AI SDK ModelMessage to our ChatMessage format for storage.
 * Tool messages can contain multiple tool results, so this returns an array.
 */
function convertModelMessageToChatMessages(msg: ModelMessage, now: string): ChatMessage[] {
  if (msg.role === "tool") {
    // Tool result - AI SDK may include multiple tool-result parts in one message.
    const content = Array.isArray(msg.content) ? msg.content : [];
    const toolMessages: ChatMessage[] = [];

    for (const part of content) {
      if (!(typeof part === "object" && part !== null && "type" in part && part.type === "tool-result")) {
        continue;
      }

      const tr = part as {
        toolCallId: string;
        toolName: string;
        output?: { type: string; value: unknown } | unknown;
        result?: unknown;
      };

      const outputContainer = tr.output ?? tr.result;
      const outputValue =
        typeof outputContainer === "object" &&
        outputContainer !== null &&
        "value" in outputContainer
          ? (outputContainer as { value: unknown }).value
          : outputContainer;

      toolMessages.push({
        id: crypto.randomUUID(),
        role: "tool",
        content:
          outputValue === undefined
            ? ""
            : typeof outputValue === "string"
              ? outputValue
              : JSON.stringify(outputValue),
        toolCallId: tr.toolCallId,
        toolName: tr.toolName,
        toolResult: outputValue,
        createdAt: now,
      });
    }

    return toolMessages;
  }

  if (msg.role === "assistant") {
    const content = msg.content;
    if (Array.isArray(content)) {
      // Extract text and tool calls - AI SDK uses 'input' not 'args'
      let textContent = "";
      const toolCalls: ChatMessage["toolCalls"] = [];

      for (const part of content) {
        if (typeof part === "object" && part !== null) {
          if ("type" in part && part.type === "text" && "text" in part) {
            textContent += (part as { text: string }).text;
          } else if ("type" in part && part.type === "tool-call") {
            const tc = part as { toolCallId: string; toolName: string; input: unknown };
            toolCalls.push({
              toolCallId: tc.toolCallId,
              toolName: tc.toolName,
              args: tc.input as Record<string, unknown>,
            });
          }
        }
      }

      return [{
        id: crypto.randomUUID(),
        role: "assistant",
        content: textContent,
        toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
        createdAt: now,
      }];
    }
    // String content
    return [{
      id: crypto.randomUUID(),
      role: "assistant",
      content: typeof content === "string" ? content : "",
      createdAt: now,
    }];
  }

  // User or other
  return [{
    id: crypto.randomUUID(),
    role: msg.role as "user" | "assistant" | "system" | "tool",
    content: typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content),
    createdAt: now,
  }];
}

/**
 * Check whether the given attachments include any images.
 */
function hasImages(attachments?: Attachment[]): boolean {
  return !!attachments?.some((a) => a.type.startsWith("image/"));
}

/**
 * Build a multimodal user message content array from text + image attachments.
 * Falls back to a plain string when there are no image attachments.
 */
async function buildUserContent(
  text: string,
  attachments?: Attachment[]
): Promise<string | UserContent> {
  if (!hasImages(attachments)) {
    return text;
  }

  const parts: UserContent = [{ type: "text", text }];

  for (const att of attachments!) {
    if (att.type.startsWith("image/") && att.path) {
      const imageData = await fs.readFile(att.path);
      parts.push({
        type: "image",
        image: imageData,
        mediaType: att.type,
      });
    }
  }

  return parts;
}

function logLLMRequest(options: {
  model: string;
  system: string;
  messages: ModelMessage[];
  toolNames: string[];
  temperature?: number;
  maxTokens?: number;
  label?: string;
}) {
  const { model, system, messages, toolNames, temperature, maxTokens, label = "LLM Request" } = options;
  console.log(`\n${LLM_LOG_BORDER}`);
  console.log(`  ${label}`);
  console.log(LLM_LOG_BORDER);
  console.log(`  Model: ${model}`);
  console.log(`  Temperature: ${temperature ?? "default"}`);
  console.log(`  Max tokens: ${maxTokens ?? "default"}`);
  console.log(`  Tools: ${toolNames.length ? toolNames.join(", ") : "none"}`);
  console.log(`  Messages: ${messages.length}`);
  console.log(LLM_LOG_BORDER);
  console.log("  --- SYSTEM ---\n");
  console.log(system);
  console.log("\n  --- MESSAGES ---");
  for (let i = 0; i < messages.length; i++) {
    const m = messages[i];
    const role = m.role.toUpperCase();
    const content = typeof m.content === "string" ? m.content : JSON.stringify(m.content);
    const preview = content.length > 500 ? content.slice(0, 500) + "…" : content;
    console.log(`  [${i + 1}] ${role}:\n${preview}`);
  }
  console.log(`\n${LLM_LOG_BORDER}\n`);
}

/**
 * Run the agent for a given chat context and return a streamable result.
 * Uses Vercel AI SDK's streamText with stopWhen for automatic tool loop.
 */
export async function runAgent(options: {
  chatId: string;
  userMessage: string;
  projectId?: string;
  currentPath?: string;
  agentNumber?: number;
  attachments?: Attachment[];
}) {
  const settings = await getSettings();
  const modelConfig = hasImages(options.attachments)
    ? settings.multimediaModel
    : settings.chatModel;
  const model = createModel(modelConfig);

  // Build context
  const context: AgentContext = {
    chatId: options.chatId,
    projectId: options.projectId,
    currentPath: options.currentPath,
    memorySubdir: options.projectId
      ? `${options.projectId}`
      : "main",
    knowledgeSubdirs: options.projectId
      ? [`${options.projectId}`, "main"]
      : ["main"],
    history: [],
    agentNumber: options.agentNumber ?? 0,
    data: {
      currentUserMessage: options.userMessage,
    },
  };

  // Load existing chat history
  const chat = await getChat(options.chatId);
  if (chat) {
    // Convert stored messages to ModelMessage format (including tool calls/results)
    context.history = convertChatMessagesToModelMessages(chat.messages);
  }

  // Build tools: base + optional MCP tools from project .meta/mcp
  const baseTools = createAgentTools(context, settings);
  let mcpCleanup: (() => Promise<void>) | undefined;
  let tools = baseTools;
  if (options.projectId) {
    const mcp = await getProjectMcpTools(options.projectId);
    if (mcp) {
      tools = { ...baseTools, ...mcp.tools };
      mcpCleanup = mcp.cleanup;
    }
  }
  tools = applyGlobalToolLoopGuard(tools);
  const toolNames = Object.keys(tools);

  // Build system prompt
  const systemPrompt = await buildSystemPrompt({
    projectId: options.projectId,
    chatId: options.chatId,
    agentNumber: options.agentNumber,
    tools: toolNames,
  });

  // Append user message to history (multimodal if image attachments present)
  const userContent = await buildUserContent(options.userMessage, options.attachments);
  const optimizedHistory = await prepareOptimizedHistory(
    context.history,
    chat,
    settings,
  );
  const messages: ModelMessage[] = [
    ...optimizedHistory,
    { role: "user", content: userContent },
  ];

  logLLMRequest({
    model: `${modelConfig.provider}/${modelConfig.model}`,
    system: systemPrompt,
    messages,
    toolNames,
    temperature: modelConfig.temperature,
    maxTokens: modelConfig.maxTokens,
    label: "LLM Request (stream)",
  });

  // Run the agent with streaming
  const result = streamText({
    model,
    system: systemPrompt,
    messages,
    tools,
    stopWhen: stepCountIs(6), // Allow up to 6 tool call rounds (reduced from 15 to prevent runaway loops)
    temperature: modelConfig.temperature ?? 0.7,
    maxOutputTokens: modelConfig.maxTokens ?? 4096,
    onFinish: async (event) => {
      if (mcpCleanup) {
        try {
          await mcpCleanup();
        } catch {
          // non-critical
        }
      }
      // Save to chat history (including tool calls and results)
      try {
        const chat = await getChat(options.chatId);
        if (chat) {
          const now = new Date().toISOString();

          // Add user message
          chat.messages.push({
            id: crypto.randomUUID(),
            role: "user",
            content: options.userMessage,
            createdAt: now,
          });

          // Add all response messages (assistant + tool calls + tool results).
          // Merge consecutive assistant-only (no tool calls) messages so that
          // multi-step agent turns don't produce duplicate text in the history.
          const responseMessages = event.response.messages;
          for (const msg of responseMessages) {
            const converted = convertModelMessageToChatMessages(msg, now);
            for (const cm of converted) {
              // If the new message is a text-only assistant message and the
              // previous stored message is also a text-only assistant message,
              // merge them to avoid duplicate bubbles in the UI.
              const prev = chat.messages.length > 0 ? chat.messages[chat.messages.length - 1] : null;
              if (
                cm.role === "assistant" &&
                prev?.role === "assistant" &&
                !cm.toolCalls?.length &&
                !prev.toolCalls?.length
              ) {
                prev.content = cm.content || prev.content;
              } else {
                chat.messages.push(cm);
              }
            }
          }

          chat.updatedAt = now;
          // Auto-title from first user message (count user messages, not total)
          const userMessageCount = chat.messages.filter(m => m.role === "user").length;
          if (userMessageCount === 1 && chat.title === "New Chat") {
            chat.title =
              options.userMessage.slice(0, 60) +
              (options.userMessage.length > 60 ? "..." : "");
          }
          await saveChat(chat);
        }
      } catch {
        // Non-critical, don't fail the response
      }

      publishUiSyncEvent({
        topic: "files",
        projectId: options.projectId ?? null,
        reason: "agent_turn_finished",
      });
    },
  });

  return result;
}

/**
 * Non-streaming agent turn for background tasks (cron/scheduler).
 */
export async function runAgentText(options: {
  chatId: string;
  userMessage: string;
  projectId?: string;
  currentPath?: string;
  agentNumber?: number;
  runtimeData?: Record<string, unknown>;
  attachments?: Attachment[];
}): Promise<string> {
  const settings = await getSettings();
  const modelConfig = hasImages(options.attachments)
    ? settings.multimediaModel
    : settings.chatModel;
  const model = createModel(modelConfig);

  const context: AgentContext = {
    chatId: options.chatId,
    projectId: options.projectId,
    currentPath: options.currentPath,
    memorySubdir: options.projectId ? `${options.projectId}` : "main",
    knowledgeSubdirs: options.projectId ? [`${options.projectId}`, "main"] : ["main"],
    history: [],
    agentNumber: options.agentNumber ?? 0,
    data: {
      ...(options.runtimeData ?? {}),
      currentUserMessage: options.userMessage,
    },
  };

  const chat = await getChat(options.chatId);
  if (chat) {
    context.history = convertChatMessagesToModelMessages(chat.messages);
  }

  const baseTools = createAgentTools(context, settings);
  let mcpCleanup: (() => Promise<void>) | undefined;
  let tools = baseTools;
  if (options.projectId) {
    const mcp = await getProjectMcpTools(options.projectId);
    if (mcp) {
      tools = { ...baseTools, ...mcp.tools };
      mcpCleanup = mcp.cleanup;
    }
  }
  tools = applyGlobalToolLoopGuard(tools);
  const toolNames = Object.keys(tools);

  const systemPrompt = await buildSystemPrompt({
    projectId: options.projectId,
    chatId: options.chatId,
    agentNumber: options.agentNumber,
    tools: toolNames,
  });

  const userContent = await buildUserContent(options.userMessage, options.attachments);
  const optimizedHistory = await prepareOptimizedHistory(
    context.history,
    chat,
    settings,
  );
  const messages: ModelMessage[] = [
    ...optimizedHistory,
    { role: "user", content: userContent },
  ];

  logLLMRequest({
    model: `${modelConfig.provider}/${modelConfig.model}`,
    system: systemPrompt,
    messages,
    toolNames,
    temperature: modelConfig.temperature,
    maxTokens: modelConfig.maxTokens,
    label: "LLM Request (non-stream)",
  });

  try {
    const generated = await generateText({
      model,
      system: systemPrompt,
      messages,
      tools,
      stopWhen: stepCountIs(6), // Reduced from 15 to prevent runaway loops
      temperature: modelConfig.temperature ?? 0.7,
      maxOutputTokens: modelConfig.maxTokens ?? 4096,
    });

    const text = extractFinalText(
      generated as unknown as { text: string; response?: { messages?: ModelMessage[] } }
    );

    try {
      const latest = await getChat(options.chatId);
      if (latest) {
        const now = new Date().toISOString();
        latest.messages.push({
          id: crypto.randomUUID(),
          role: "user",
          content: options.userMessage,
          createdAt: now,
        });

        const responseMessages = (
          generated as unknown as { response?: { messages?: ModelMessage[] } }
        ).response?.messages;

        if (Array.isArray(responseMessages) && responseMessages.length > 0) {
          for (const msg of responseMessages) {
            const converted = convertModelMessageToChatMessages(msg, now);
            for (const cm of converted) {
              const prev = latest.messages.length > 0 ? latest.messages[latest.messages.length - 1] : null;
              if (
                cm.role === "assistant" &&
                prev?.role === "assistant" &&
                !cm.toolCalls?.length &&
                !prev.toolCalls?.length
              ) {
                prev.content = cm.content || prev.content;
              } else {
                latest.messages.push(cm);
              }
            }
          }
        } else {
          latest.messages.push({
            id: crypto.randomUUID(),
            role: "assistant",
            content: text,
            createdAt: now,
          });
        }

        latest.updatedAt = now;
        await saveChat(latest);
      }
    } catch {
      // Non-critical for background runs.
    }

    publishUiSyncEvent({
      topic: "files",
      projectId: options.projectId ?? null,
      reason: "agent_turn_finished",
    });

    return text;
  } finally {
    if (mcpCleanup) {
      try {
        await mcpCleanup();
      } catch {
        // non-critical
      }
    }
  }
}

/**
 * Run agent for subordinate delegation (non-streaming, returns result)
 */
export async function runSubordinateAgent(options: {
  task: string;
  projectId?: string;
  parentAgentNumber: number;
  parentHistory: ModelMessage[];
}): Promise<string> {
  const settings = await getSettings();
  const model = createModel(settings.utilityModel);

  const context: AgentContext = {
    chatId: `subordinate-${Date.now()}`,
    projectId: options.projectId,
    memorySubdir: options.projectId
      ? `projects/${options.projectId}`
      : "main",
    knowledgeSubdirs: options.projectId
      ? [`projects/${options.projectId}`, "main"]
      : ["main"],
    history: [],
    agentNumber: options.parentAgentNumber + 1,
    data: {},
  };

  let tools = createAgentTools(context, settings);
  let mcpCleanupSub: (() => Promise<void>) | undefined;
  if (options.projectId) {
    const mcp = await getProjectMcpTools(options.projectId);
    if (mcp) {
      tools = { ...tools, ...mcp.tools };
      mcpCleanupSub = mcp.cleanup;
    }
  }
  tools = applyGlobalToolLoopGuard(tools);
  const toolNames = Object.keys(tools);

  const systemPrompt = await buildSystemPrompt({
    projectId: options.projectId,
    agentNumber: context.agentNumber,
    tools: toolNames,
  });

  // Include relevant parent history for context
  const relevantHistory = trimHistory(options.parentHistory).slice(-6);

  const messages: ModelMessage[] = [
    ...relevantHistory,
    {
      role: "user",
      content: `You are a subordinate agent. Complete this task and report back:\n\n${options.task}`,
    },
  ];

  logLLMRequest({
    model: `${settings.utilityModel.provider}/${settings.utilityModel.model}`,
    system: systemPrompt,
    messages,
    toolNames,
    temperature: settings.utilityModel.temperature,
    maxTokens: settings.utilityModel.maxTokens,
    label: "LLM Request (subordinate)",
  });

  try {
    const { text } = await generateText({
      model,
      system: systemPrompt,
      messages,
      tools,
      stopWhen: stepCountIs(5), // Reduced from 10 to prevent runaway loops
      temperature: settings.utilityModel.temperature ?? 0.7,
      maxOutputTokens: settings.utilityModel.maxTokens ?? 4096,
    });
    return text;
  } finally {
    if (mcpCleanupSub) {
      try {
        await mcpCleanupSub();
      } catch {
        // non-critical
      }
    }
  }
}
