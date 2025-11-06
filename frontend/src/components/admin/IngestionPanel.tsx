import { useState, useEffect, useRef } from "react";
import { Upload, Loader2, CheckCircle, XCircle, Clock, RefreshCw, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { 
  uploadAndIngestFile, 
  getIngestionStatus, 
  listIngestionTasks, 
  cancelIngestionTask,
  type IngestionTask 
} from "@/api/client";

const SOURCE_TYPES = [
  { value: "quran", label: "Quran" },
  { value: "hadith", label: "Hadith" },
  { value: "tafsir", label: "Tafsir" },
  { value: "fiqh", label: "Fiqh" },
  { value: "seerah", label: "Seerah" },
  { value: "aqidah", label: "Aqidah" },
];

export default function IngestionPanel() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [sourceType, setSourceType] = useState("quran");
  const [collectionName, setCollectionName] = useState("");
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  
  // Current task being tracked
  const [currentTask, setCurrentTask] = useState<IngestionTask | null>(null);
  const [taskHistory, setTaskHistory] = useState<IngestionTask[]>([]);
  
  // Polling interval ref
  const pollingRef = useRef<NodeJS.Timeout | null>(null);

  // Fetch task history on mount
  useEffect(() => {
    fetchTaskHistory();
  }, []);

  // Poll for status updates when there's a current task
  useEffect(() => {
    if (currentTask && (currentTask.status === "queued" || currentTask.status === "running")) {
      // Start polling
      pollingRef.current = setInterval(() => {
        pollTaskStatus(currentTask.task_id);
      }, 2000); // Poll every 2 seconds

      // Cleanup on unmount or when task completes
      return () => {
        if (pollingRef.current) {
          clearInterval(pollingRef.current);
          pollingRef.current = null;
        }
      };
    }
  }, [currentTask?.task_id, currentTask?.status]);

  const fetchTaskHistory = async () => {
    try {
      const tasks = await listIngestionTasks(10);
      setTaskHistory(tasks);
    } catch (error) {
      console.error("Failed to fetch task history:", error);
    }
  };

  const pollTaskStatus = async (taskId: string) => {
    try {
      const status = await getIngestionStatus(taskId);
      setCurrentTask(status);

      // If task completed or failed, refresh history
      if (status.status === "completed" || status.status === "failed") {
        if (pollingRef.current) {
          clearInterval(pollingRef.current);
          pollingRef.current = null;
        }
        fetchTaskHistory();
      }
    } catch (error) {
      console.error("Failed to poll task status:", error);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setUploadError(null);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    setUploading(true);
    setUploadError(null);

    try {
      const response = await uploadAndIngestFile(
        selectedFile,
        sourceType,
        collectionName || undefined
      );

      if (!response.success) {
        setUploadError(response.error || "Upload failed");
        setUploading(false);
        return;
      }

      // Start tracking this task
      const initialTask: IngestionTask = {
        task_id: response.task_id,
        status: "queued",
        progress: 0,
        message: response.message,
        file_name: response.file_name,
        collection_name: response.collection_name,
        created_at: new Date().toISOString(),
      };

      setCurrentTask(initialTask);
      setSelectedFile(null);
      setUploading(false);

      // Reset file input
      const fileInput = document.getElementById("file-input") as HTMLInputElement;
      if (fileInput) fileInput.value = "";

      // Start polling immediately
      pollTaskStatus(response.task_id);
    } catch (error) {
      console.error("Upload error:", error);
      setUploadError("Failed to upload file. Please check that the backend is running.");
      setUploading(false);
    }
  };

  const handleCancelTask = async (taskId: string) => {
    try {
      await cancelIngestionTask(taskId);
      if (currentTask?.task_id === taskId) {
        setCurrentTask(null);
      }
      fetchTaskHistory();
    } catch (error) {
      console.error("Failed to cancel task:", error);
    }
  };

  const formatDuration = (seconds: number) => {
    if (seconds < 60) return `${seconds.toFixed(1)}s`;
    if (seconds < 3600) return `${(seconds / 60).toFixed(1)}m`;
    return `${(seconds / 3600).toFixed(1)}h`;
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case "completed":
        return "text-islamic-emerald-600 bg-islamic-emerald-50 border-islamic-emerald-200";
      case "failed":
        return "text-red-600 bg-red-50 border-red-200";
      case "running":
        return "text-blue-600 bg-blue-50 border-blue-200";
      case "queued":
        return "text-yellow-600 bg-yellow-50 border-yellow-200";
      default:
        return "text-gray-600 bg-gray-50 border-gray-200";
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "completed":
        return <CheckCircle className="w-5 h-5" />;
      case "failed":
        return <XCircle className="w-5 h-5" />;
      case "running":
        return <Loader2 className="w-5 h-5 animate-spin" />;
      case "queued":
        return <Clock className="w-5 h-5" />;
      default:
        return <Clock className="w-5 h-5" />;
    }
  };

  return (
    <div className="space-y-6">
      {/* Upload Form */}
      <Card>
        <CardHeader>
          <CardTitle>Upload & Ingest Data</CardTitle>
          <CardDescription>
            Upload JSON files containing Islamic texts for ingestion into the knowledge base
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium mb-2">Source Type</label>
              <select
                value={sourceType}
                onChange={(e) => setSourceType(e.target.value)}
                disabled={uploading || currentTask?.status === "running"}
                className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-50"
              >
                {SOURCE_TYPES.map((type) => (
                  <option key={type.value} value={type.value}>
                    {type.label}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">
                Collection Name <span className="text-muted-foreground">(optional)</span>
              </label>
              <Input
                value={collectionName}
                onChange={(e) => setCollectionName(e.target.value)}
                placeholder="Use default collection"
                disabled={uploading || currentTask?.status === "running"}
              />
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium mb-2">JSON File</label>
            <Input
              id="file-input"
              type="file"
              accept=".json"
              onChange={handleFileChange}
              disabled={uploading || currentTask?.status === "running"}
            />
            {selectedFile && (
              <p className="text-sm text-muted-foreground mt-2">
                Selected: {selectedFile.name} ({(selectedFile.size / 1024 / 1024).toFixed(2)} MB)
              </p>
            )}
          </div>

          {uploadError && (
            <div className="p-3 rounded-md bg-red-50 border border-red-200 flex items-start gap-2">
              <XCircle className="w-4 h-4 text-red-600 flex-shrink-0 mt-0.5" />
              <p className="text-sm text-red-600">{uploadError}</p>
            </div>
          )}

          <Button
            onClick={handleUpload}
            disabled={!selectedFile || uploading || currentTask?.status === "running"}
            className="w-full"
          >
            {uploading ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Uploading...
              </>
            ) : (
              <>
                <Upload className="w-4 h-4 mr-2" />
                Upload & Ingest
              </>
            )}
          </Button>
        </CardContent>
      </Card>

      {/* Current Task Progress */}
      {currentTask && (currentTask.status === "queued" || currentTask.status === "running") && (
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="text-base">Current Ingestion</CardTitle>
              {currentTask.status === "queued" && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => handleCancelTask(currentTask.task_id)}
                >
                  <X className="w-4 h-4 mr-1" />
                  Cancel
                </Button>
              )}
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div className="flex items-start gap-3">
                <div className={`p-2 rounded-md ${getStatusColor(currentTask.status)}`}>
                  {getStatusIcon(currentTask.status)}
                </div>
                <div className="flex-1">
                  <p className="font-medium">{currentTask.file_name}</p>
                  <p className="text-sm text-muted-foreground">{currentTask.message}</p>
                  {currentTask.queue_position && (
                    <p className="text-sm text-muted-foreground mt-1">
                      Position in queue: {currentTask.queue_position}
                    </p>
                  )}
                </div>
              </div>

              {currentTask.status === "running" && (
                <div>
                  <div className="flex justify-between text-sm mb-2">
                    <span>Progress</span>
                    <span className="font-medium">{currentTask.progress.toFixed(1)}%</span>
                  </div>
                  <div className="bg-gray-200 rounded-full h-2 overflow-hidden">
                    <div
                      className="bg-islamic-emerald-500 h-full transition-all duration-300"
                      style={{ width: `${currentTask.progress}%` }}
                    />
                  </div>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Task History */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-base">Ingestion History</CardTitle>
              <CardDescription>Recent ingestion tasks</CardDescription>
            </div>
            <Button variant="outline" size="sm" onClick={fetchTaskHistory}>
              <RefreshCw className="w-4 h-4" />
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {taskHistory.length === 0 ? (
            <p className="text-sm text-muted-foreground text-center py-4">
              No ingestion tasks yet
            </p>
          ) : (
            <div className="space-y-3">
              {taskHistory.map((task) => (
                <div
                  key={task.task_id}
                  className={`p-3 rounded-md border ${getStatusColor(task.status)}`}
                >
                  <div className="flex items-start gap-3">
                    <div className="flex-shrink-0 mt-0.5">
                      {getStatusIcon(task.status)}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-start justify-between gap-2">
                        <div className="flex-1">
                          <p className="font-medium text-sm truncate">{task.file_name}</p>
                          <p className="text-xs opacity-75 mt-0.5">{task.message}</p>
                        </div>
                        <span className="text-xs opacity-75 whitespace-nowrap">
                          {new Date(task.created_at).toLocaleString()}
                        </span>
                      </div>

                      {task.result && (
                        <div className="mt-2 text-xs space-y-0.5">
                          <p>Documents: {task.result.documents_processed}</p>
                          <p>Nodes: {task.result.nodes_created}</p>
                          <p>Points in DB: {task.result.total_points}</p>
                          <p>Duration: {formatDuration(task.result.time_elapsed)}</p>
                        </div>
                      )}

                      {task.error && (
                        <p className="mt-2 text-xs font-medium">Error: {task.error}</p>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* File Format Guidelines */}
      <Card className="bg-muted/30">
        <CardHeader>
          <CardTitle className="text-base">File Format Guidelines</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-sm space-y-2 text-muted-foreground">
            <p>• Files must be in JSON format</p>
            <p>• Each source type has a specific schema requirement</p>
            <p>• Quran: verses with surah_number, verse_number, arabic_text, english_text</p>
            <p>• Hadith: narrator_chain, arabic_text, english_text, authenticity_grade</p>
            <p>• Tafsir: verse_key, commentary_text, scholar_name</p>
            <p>• Processing happens in the background - you can close this page</p>
            <p>• Only one ingestion runs at a time; additional uploads are queued</p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
