import { useState, useEffect } from "react";
import { Database, Trash2, RefreshCw, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { listCollections, deleteCollection, clearCollection } from "@/api/client";
import type { CollectionInfo } from "@/api/client";

export default function CollectionManager() {
  const [collections, setCollections] = useState<CollectionInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState<string | null>(null);

  useEffect(() => {
    fetchCollections();
  }, []);

  const fetchCollections = async () => {
    setLoading(true);
    try {
      const data = await listCollections();
      setCollections(data);
    } catch (error) {
      console.error("Error fetching collections:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleClearCollection = async (collectionName: string) => {
    if (!confirm(`Are you sure you want to clear all data from ${collectionName}?`)) {
      return;
    }

    setActionLoading(collectionName + "-clear");
    try {
      await clearCollection(collectionName);
      await fetchCollections();
    } catch (error) {
      console.error("Error clearing collection:", error);
      alert("Failed to clear collection");
    } finally {
      setActionLoading(null);
    }
  };

  const handleDeleteCollection = async (collectionName: string) => {
    if (!confirm(`Are you sure you want to permanently delete ${collectionName}?`)) {
      return;
    }

    setActionLoading(collectionName + "-delete");
    try {
      await deleteCollection(collectionName);
      await fetchCollections();
    } catch (error) {
      console.error("Error deleting collection:", error);
      alert("Failed to delete collection");
    } finally {
      setActionLoading(null);
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
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <div>
          <h3 className="text-lg font-semibold heading-elegant">Qdrant Collections</h3>
          <p className="text-sm text-muted-foreground">
            Manage vector database collections • localhost:6333
          </p>
        </div>
        <Button onClick={fetchCollections} variant="outline" size="sm">
          <RefreshCw className="w-4 h-4 mr-2" />
          Refresh
        </Button>
      </div>

      {collections.length === 0 ? (
        <Card className="islamic-card-border">
          <CardContent className="py-12">
            <div className="text-center text-muted-foreground">
              <Database className="w-12 h-12 mx-auto mb-4 opacity-50" />
              <p>No collections found</p>
              <p className="text-sm mt-2">Upload and ingest data to create collections</p>
            </div>
          </CardContent>
        </Card>
      ) : (
        <div className="grid gap-4">
          {collections.map((collection) => (
            <Card key={collection.name} className="islamic-card-border">
              <CardHeader>
                <div className="flex items-start justify-between">
                  <div>
                    <CardTitle className="text-base">{collection.name}</CardTitle>
                    <CardDescription>
                      {collection.points_count || 0} points •{" "}
                      {collection.vectors_count || 0} vectors
                    </CardDescription>
                  </div>
                  <div className="flex gap-2">
                    <Button
                      onClick={() => handleClearCollection(collection.name)}
                      variant="outline"
                      size="sm"
                      disabled={actionLoading !== null}
                    >
                      {actionLoading === collection.name + "-clear" ? (
                        <Loader2 className="w-4 h-4 animate-spin" />
                      ) : (
                        <>
                          <RefreshCw className="w-4 h-4 mr-2" />
                          Clear
                        </>
                      )}
                    </Button>
                    <Button
                      onClick={() => handleDeleteCollection(collection.name)}
                      variant="destructive"
                      size="sm"
                      disabled={actionLoading !== null}
                    >
                      {actionLoading === collection.name + "-delete" ? (
                        <Loader2 className="w-4 h-4 animate-spin" />
                      ) : (
                        <>
                          <Trash2 className="w-4 h-4 mr-2" />
                          Delete
                        </>
                      )}
                    </Button>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <p className="text-muted-foreground">Vector Size</p>
                    <p className="font-medium">
                      {collection.config?.params?.vectors?.size || "N/A"}
                    </p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Distance</p>
                    <p className="font-medium">
                      {collection.config?.params?.vectors?.distance || "N/A"}
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}

