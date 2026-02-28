import { AppSidebar } from "@/components/app-sidebar"
import { SiteHeader } from "@/components/site-header"
import { ChatPanel } from "@/components/chat/chat-panel"
import {
  SidebarInset,
  SidebarProvider,
} from "@/components/ui/sidebar"
import { redirect } from "next/navigation"
import {
  getAllProjects,
} from "@/lib/storage/project-store"

export const dynamic = "force-dynamic"

export default async function DashboardPage() {
  const projects = await getAllProjects()

  if (projects.length === 0) {
    redirect("/dashboard/projects")
  }

  return (
    <div className="h-dvh flex flex-col overflow-hidden [--header-height:calc(--spacing(14))]">
      <SidebarProvider className="flex flex-col h-full overflow-hidden">
        <SiteHeader title="Chat" />
        <div className="flex flex-1 min-h-0 overflow-hidden">
          <AppSidebar />
          <SidebarInset>
            <div className="flex flex-1 flex-col min-h-0 h-full overflow-hidden">
              <ChatPanel />
            </div>
          </SidebarInset>
        </div>
      </SidebarProvider>
    </div>
  )
}
