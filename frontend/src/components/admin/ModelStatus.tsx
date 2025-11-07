import { useState, useEffect } from "react";
import { CheckCircle, XCircle, Download, Loader2, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { listModels, pullModel, checkHealth } from "@/api/client";
import type { ModelInfo } from "@/api/client";

export default function ModelStatus() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [health, setHealth] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [pullingModel, setPullingModel] = useState<string | null>(null);

  useEffect(() => {
    fetchStatus();
  }, []);

  const fetchStatus = async () => {
    setLoading(true);
    try {
      const [modelsData, healthData] = await Promise.all([
        listModels(),
        checkHealth(),
      ]);
      setModels(modelsData);
      setHealth(healthData);
    } catch (error) {
      console.error("Error fetching status:", error);
    } finally {
      setLoading(false);
    }
  };

  const handlePullModel = async (modelName: string) => {
    setPullingModel(modelName);
    try {
      await pullModel(modelName);
      await fetchStatus();
    } catch (error) {
      console.error("Error pulling model:", error);
      alert("Failed to pull model");
    } finally {
      setPullingModel(null);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="w-8 h-8 animate-spin text-islamic-emerald-500" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Health Status */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-base">System Health</CardTitle>
              <CardDescription>Backend services status</CardDescription>
            </div>
            <Button onClick={fetchStatus} variant="outline" size="sm">
              <RefreshCw className="w-4 h-4 mr-2" />
              Refresh
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="flex items-center gap-3 p-3 rounded-lg bg-muted/50 islamic-card-border">
              {health?.services?.ollama?.status === "healthy" ? (
                <CheckCircle className="w-5 h-5 text-primary" />
              ) : (
                <XCircle className="w-5 h-5 text-red-500" />
              )}
              <div>
                <p className="text-sm font-medium">Ollama</p>
                <p className="text-xs text-muted-foreground">
                  {health?.services?.ollama?.status === "healthy" ? "Running :11434" : "Offline"}
                </p>
              </div>
            </div>

            <div className="flex items-center gap-3 p-3 rounded-lg bg-muted/50 islamic-card-border">
              {health?.services?.lmstudio?.status === "healthy" ? (
                <CheckCircle className="w-5 h-5 text-primary" />
              ) : (
                <XCircle className="w-5 h-5 text-red-500" />
              )}
              <div>
                <p className="text-sm font-medium">LM Studio</p>
                <p className="text-xs text-muted-foreground">
                  {health?.services?.lmstudio?.status === "healthy" ? "Running :1234" : "Offline"}
                </p>
              </div>
            </div>

            <div className="flex items-center gap-3 p-3 rounded-lg bg-muted/50 islamic-card-border">
              {health?.services?.qdrant?.status === "healthy" ? (
                <CheckCircle className="w-5 h-5 text-primary" />
              ) : (
                <XCircle className="w-5 h-5 text-red-500" />
              )}
              <div>
                <p className="text-sm font-medium">Qdrant</p>
                <p className="text-xs text-muted-foreground">
                  {health?.services?.qdrant?.status === "healthy" ? "Running :6333" : "Offline"}
                </p>
              </div>
            </div>

            <div className="flex items-center gap-3 p-3 rounded-lg bg-muted/50 islamic-card-border">
              {health?.services?.langgraph?.status === "healthy" ? (
                <CheckCircle className="w-5 h-5 text-primary" />
              ) : (
                <XCircle className="w-5 h-5 text-red-500" />
              )}
              <div>
                <p className="text-sm font-medium">LangGraph</p>
                <p className="text-xs text-muted-foreground">
                  {health?.services?.langgraph?.status === "healthy" ? "Running :8123" : "Offline"}
                </p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Ollama Models */}
      <div>
        <h3 className="text-lg font-semibold mb-4">Ollama Models</h3>
        <div className="grid gap-4">
          {models.length === 0 ? (
            <Card>
              <CardContent className="py-12">
                <div className="text-center text-muted-foreground">
                  <p>No models found</p>
                  <p className="text-sm mt-2">Pull models using Ollama CLI</p>
                </div>
              </CardContent>
            </Card>
          ) : (
            models.map((model) => (
              <Card key={model.name}>
                <CardHeader>
                  <div className="flex items-start justify-between">
                    <div>
                      <CardTitle className="text-base flex items-center gap-2">
                        {model.name}
                        {model.loaded ? (
                          <CheckCircle className="w-4 h-4 text-islamic-emerald-500" />
                        ) : (
                          <XCircle className="w-4 h-4 text-muted-foreground" />
                        )}
                      </CardTitle>
                      <CardDescription>
                        {(model.size / 1024 / 1024 / 1024).toFixed(2)} GB
                      </CardDescription>
                    </div>
                    {!model.loaded && (
                      <Button
                        onClick={() => handlePullModel(model.name)}
                        variant="outline"
                        size="sm"
                        disabled={pullingModel !== null}
                      >
                        {pullingModel === model.name ? (
                          <Loader2 className="w-4 h-4 animate-spin" />
                        ) : (
                          <>
                            <Download className="w-4 h-4 mr-2" />
                            Pull
                          </>
                        )}
                      </Button>
                    )}
                  </div>
                </CardHeader>
              </Card>
            ))
          )}
        </div>
      </div>

      {/* Service Information */}
      <Card className="bg-muted/30 islamic-card-border">
        <CardHeader>
          <CardTitle className="text-base">Service Information</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-sm space-y-3 text-muted-foreground">
            <div>
              <p className="font-medium text-foreground mb-1">LLM Services:</p>
              <p>• <strong>Ollama:</strong> localhost:11434 (Docker)</p>
              <p>• <strong>LM Studio:</strong> localhost:1234 (Local)</p>
            </div>
            <div>
              <p className="font-medium text-foreground mb-1">Vector Database:</p>
              <p>• <strong>Qdrant:</strong> localhost:6333 (HTTP), localhost:6334 (gRPC)</p>
            </div>
            <div>
              <p className="font-medium text-foreground mb-1">Recommended Models:</p>
              <p>• <strong>Embeddings:</strong> nomic-embed-text:latest</p>
              <p>• <strong>Chat LLM:</strong> qwen2.5:3b or llama3.2:3b</p>
              <p className="mt-2">Pull models: <code className="px-1.5 py-0.5 bg-muted rounded text-primary">ollama pull model-name</code></p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

