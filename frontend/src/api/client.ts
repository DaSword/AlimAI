import { Client } from "@langchain/langgraph-sdk";
import axios from "axios";

// LangGraph Server configuration
const LANGGRAPH_URL = import.meta.env.VITE_LANGGRAPH_URL || "http://localhost:8123";
const GRAPH_NAME = "rag_assistant"; // Name defined in backend/api/server.py

// LangGraph SDK client for chat operations
export const langgraphClient = new Client({
  apiUrl: LANGGRAPH_URL,
});

// Admin API client for management operations
export const adminClient = axios.create({
  baseURL: `${LANGGRAPH_URL}/api/admin`,
  headers: {
    "Content-Type": "application/json",
  },
});

// Assistant ID cache
let cachedAssistantId: string | null = null;

// Chat API types
export interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
}

export interface StreamEvent {
  event: string;
  data: any;
}

// Chat API functions
export async function getAssistantId(): Promise<string> {
  if (cachedAssistantId) {
    return cachedAssistantId;
  }

  try {
    const assistants = await langgraphClient.assistants.search({
      metadata: null,
      offset: 0,
      limit: 10,
    });

    // Find the assistant with the matching graph name
    const assistant = assistants.find((a) => a.graph_id === GRAPH_NAME);
    
    if (!assistant) {
      throw new Error(`Assistant with graph_id '${GRAPH_NAME}' not found`);
    }

    cachedAssistantId = assistant.assistant_id;
    return cachedAssistantId;
  } catch (error) {
    console.error("Error fetching assistant:", error);
    throw error;
  }
}

export async function createThread(): Promise<string> {
  try {
    const thread = await langgraphClient.threads.create();
    return thread.thread_id;
  } catch (error) {
    console.error("Error creating thread:", error);
    throw error;
  }
}

export async function* streamChatResponse(
  threadId: string,
  message: string,
  timeoutMs: number = 180000 // 3 minutes timeout (default)
): AsyncGenerator<StreamEvent> {
  let streamIterator: AsyncIterator<any> | null = null;
  let hasTimedOut = false;
  let lastEventTime = Date.now();

  // Create a timeout checker
  const timeoutId = setTimeout(() => {
    hasTimedOut = true;
    console.warn(`Stream timeout after ${timeoutMs}ms without completion`);
  }, timeoutMs);

  try {
    // Get the assistant ID (cached after first call)
    const assistantId = await getAssistantId();
    
    const stream = langgraphClient.runs.stream(
      threadId,
      assistantId,
      {
        input: {
          messages: [{ role: "user", content: message }],
        },
      }
    );

    streamIterator = stream[Symbol.asyncIterator]();

    while (true) {
      // Check for timeout before each iteration
      if (hasTimedOut) {
        throw new Error(`Request timeout: The response took longer than ${timeoutMs / 1000} seconds. Please try again or simplify your question.`);
      }

      lastEventTime = Date.now();
      const result = await streamIterator.next();
      if (result.done) {
        break;
      }
      yield result.value;
    }
  } finally {
    clearTimeout(timeoutId);
  }
}

export async function getThreadHistory(threadId: string): Promise<ChatMessage[]> {
  try {
    const state = await langgraphClient.threads.getState(threadId);
    return state.values.messages || [];
  } catch (error) {
    console.error("Error getting thread history:", error);
    return [];
  }
}

// Admin API types
export interface CollectionInfo {
  name: string;
  vectors_count: number;
  points_count: number;
  config: any;
}

export interface ModelInfo {
  name: string;
  size: number;
  loaded: boolean;
}

export interface IngestionProgress {
  status: "processing" | "completed" | "error";
  progress: number;
  message: string;
}

// Admin API functions

// Ingestion endpoints
export async function uploadAndIngestFile(
  file: File,
  sourceType: string
): Promise<IngestionProgress> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("source_type", sourceType);

  const response = await adminClient.post("/ingestion/upload", formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });
  return response.data;
}

export async function getIngestionStatus(taskId: string): Promise<IngestionProgress> {
  const response = await adminClient.get(`/ingestion/status/${taskId}`);
  return response.data;
}

// Collection endpoints
export async function listCollections(): Promise<CollectionInfo[]> {
  const response = await adminClient.get("/collections");
  return response.data;
}

export async function getCollectionInfo(collectionName: string): Promise<CollectionInfo> {
  const response = await adminClient.get(`/collections/${collectionName}`);
  return response.data;
}

export async function deleteCollection(collectionName: string): Promise<void> {
  await adminClient.delete(`/collections/${collectionName}`);
}

export async function clearCollection(collectionName: string): Promise<void> {
  await adminClient.post(`/collections/${collectionName}/clear`);
}

// Model endpoints
export async function listModels(): Promise<ModelInfo[]> {
  const response = await adminClient.get("/models");
  return response.data;
}

export async function getModelStatus(modelName: string): Promise<ModelInfo> {
  const response = await adminClient.get(`/models/${modelName}`);
  return response.data;
}

export async function pullModel(modelName: string): Promise<void> {
  await adminClient.post(`/models/${modelName}/pull`);
}

export async function checkHealth(): Promise<{ status: string; services: any }> {
  const response = await adminClient.get("/health");
  return response.data;
}

