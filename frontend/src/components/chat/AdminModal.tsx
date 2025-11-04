import { X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import IngestionPanel from "@/components/admin/IngestionPanel";
import CollectionManager from "@/components/admin/CollectionManager";
import ModelStatus from "@/components/admin/ModelStatus";

interface AdminModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export function AdminModal({ isOpen, onClose }: AdminModalProps) {
  if (!isOpen) return null;

  return (
    <div 
      className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4"
      onClick={(e) => {
        if (e.target === e.currentTarget) {
          onClose();
        }
      }}
    >
      <div className="bg-card rounded-2xl shadow-2xl max-w-5xl w-full max-h-[85vh] overflow-hidden animate-fadeIn islamic-card-border">
        {/* Modal Header */}
        <div className="flex items-center justify-between p-6 border-b border-border">
          <h2 className="text-2xl font-semibold heading-elegant">Admin Tools</h2>
          <Button
            variant="ghost"
            size="sm"
            onClick={onClose}
            className="h-8 w-8 p-0"
          >
            <X className="w-4 h-4" />
          </Button>
        </div>

        {/* Modal Content */}
        <div className="overflow-y-auto max-h-[calc(85vh-80px)] custom-scrollbar">
          <div className="p-6">
            <Card className="p-6 islamic-card-border">
              <Tabs defaultValue="ingestion">
                <TabsList className="grid w-full grid-cols-3">
                  <TabsTrigger value="ingestion">Ingestion</TabsTrigger>
                  <TabsTrigger value="collections">Collections</TabsTrigger>
                  <TabsTrigger value="models">Models</TabsTrigger>
                </TabsList>

                <TabsContent value="ingestion" className="mt-6">
                  <IngestionPanel />
                </TabsContent>

                <TabsContent value="collections" className="mt-6">
                  <CollectionManager />
                </TabsContent>

                <TabsContent value="models" className="mt-6">
                  <ModelStatus />
                </TabsContent>
              </Tabs>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}

