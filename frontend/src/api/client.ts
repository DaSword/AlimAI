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
  message: string
): AsyncGenerator<StreamEvent> {
  try {
    const stream = langgraphClient.runs.stream(
      threadId,
      GRAPH_NAME,
      {
        input: {
          messages: [{ role: "user", content: message }],
        },
      }
    );

    for await (const event of stream) {
      yield event;
    }
  } catch (error) {
    console.error("Error streaming chat response:", error);
    throw error;
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

