import { useState } from "react";
import { Upload, Loader2, CheckCircle, XCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { uploadAndIngestFile } from "@/api/client";

const SOURCE_TYPES = [
  { value: "quran", label: "Quran" },
  { value: "hadith", label: "Hadith" },
  { value: "tafsir", label: "Tafsir" },
  { value: "fiqh", label: "Fiqh" },
  { value: "seerah", label: "Seerah" },
  { value: "aqidah", label: "Aqidah" },
];

interface UploadStatus {
  status: "idle" | "uploading" | "processing" | "completed" | "error";
  message: string;
  progress?: number;
}

export default function IngestionPanel() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [sourceType, setSourceType] = useState("quran");
  const [uploadStatus, setUploadStatus] = useState<UploadStatus>({
    status: "idle",
    message: "",
  });

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setUploadStatus({ status: "idle", message: "" });
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    setUploadStatus({
      status: "uploading",
      message: "Uploading file...",
      progress: 0,
    });

    try {
      const result = await uploadAndIngestFile(selectedFile, sourceType);

      if (result.status === "completed") {
        setUploadStatus({
          status: "completed",
          message: `Successfully ingested ${selectedFile.name}`,
        });
        setSelectedFile(null);
        // Reset file input
        const fileInput = document.getElementById("file-input") as HTMLInputElement;
        if (fileInput) fileInput.value = "";
      } else if (result.status === "error") {
        setUploadStatus({
          status: "error",
          message: result.message || "Failed to ingest file",
        });
      } else {
        setUploadStatus({
          status: "processing",
          message: result.message || "Processing...",
          progress: result.progress,
        });
      }
    } catch (error) {
      console.error("Upload error:", error);
      setUploadStatus({
        status: "error",
        message: "Failed to upload file. Please check that the backend is running.",
      });
    }
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Upload & Ingest Data</CardTitle>
          <CardDescription>
            Upload JSON files containing Islamic texts for ingestion into the knowledge base
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-2">Source Type</label>
            <select
              value={sourceType}
              onChange={(e) => setSourceType(e.target.value)}
              className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            >
              {SOURCE_TYPES.map((type) => (
                <option key={type.value} value={type.value}>
                  {type.label}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium mb-2">JSON File</label>
            <Input
              id="file-input"
              type="file"
              accept=".json"
              onChange={handleFileChange}
              disabled={uploadStatus.status === "uploading" || uploadStatus.status === "processing"}
            />
            {selectedFile && (
              <p className="text-sm text-muted-foreground mt-2">
                Selected: {selectedFile.name} ({(selectedFile.size / 1024).toFixed(2)} KB)
              </p>
            )}
          </div>

          <Button
            onClick={handleUpload}
            disabled={!selectedFile || uploadStatus.status === "uploading" || uploadStatus.status === "processing"}
            className="w-full"
          >
            {uploadStatus.status === "uploading" || uploadStatus.status === "processing" ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                {uploadStatus.message}
              </>
            ) : (
              <>
                <Upload className="w-4 h-4 mr-2" />
                Upload & Ingest
              </>
            )}
          </Button>

          {uploadStatus.status !== "idle" && (
            <div
              className={`p-4 rounded-md flex items-start gap-3 ${
                uploadStatus.status === "completed"
                  ? "bg-islamic-emerald-50 border border-islamic-emerald-200"
                  : uploadStatus.status === "error"
                  ? "bg-red-50 border border-red-200"
                  : "bg-blue-50 border border-blue-200"
              }`}
            >
              {uploadStatus.status === "completed" && (
                <CheckCircle className="w-5 h-5 text-islamic-emerald-600 flex-shrink-0 mt-0.5" />
              )}
              {uploadStatus.status === "error" && (
                <XCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
              )}
              {(uploadStatus.status === "uploading" || uploadStatus.status === "processing") && (
                <Loader2 className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5 animate-spin" />
              )}
              <div className="flex-1">
                <p className="text-sm font-medium">{uploadStatus.message}</p>
                {uploadStatus.progress !== undefined && (
                  <div className="mt-2 bg-white rounded-full h-2 overflow-hidden">
                    <div
                      className="bg-islamic-emerald-500 h-full transition-all duration-300"
                      style={{ width: `${uploadStatus.progress}%` }}
                    />
                  </div>
                )}
              </div>
            </div>
          )}
        </CardContent>
      </Card>

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
            <p>• Refer to the documentation for complete schema details</p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

