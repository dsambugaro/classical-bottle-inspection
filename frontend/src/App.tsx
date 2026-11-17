import { lazy, Suspense, useEffect } from "react";
import { useTranslation } from "react-i18next";
import { createBrowserRouter, Outlet } from "react-router";
import { Toaster } from "sonner";

import { Shell } from "./components/Shell";
import { Spinner, TooltipProvider } from "./components/ui/primitives";
import { useRuleAlerts } from "./lib/alerts";
import { rememberLanguage } from "./lib/i18n";
import { useGlobalEventInvalidation, useSettings } from "./lib/queries";
import { InspectPage } from "./pages/InspectPage";

const ProfilesPage = lazy(() => import("./pages/ProfilesPage"));
const ReportsPage = lazy(() => import("./pages/ReportsPage"));
const IntegrationsPage = lazy(() => import("./pages/IntegrationsPage"));
const SystemPage = lazy(() => import("./pages/SystemPage"));
const DebugPage = lazy(() => import("./pages/DebugPage"));

/** Applies settings.language when the browser has no stored choice yet (REQ-L1). */
function useLanguageFromSettings(): void {
  const { i18n } = useTranslation();
  const settings = useSettings().data;
  useEffect(() => {
    let stored: string | null;
    try {
      stored = localStorage.getItem("bottle-inspection.language");
    } catch {
      stored = null;
    }
    if (!stored && settings?.language && settings.language !== i18n.language) {
      void i18n.changeLanguage(settings.language);
      rememberLanguage(settings.language);
    }
  }, [settings?.language, i18n]);
}

function Layout() {
  useGlobalEventInvalidation();
  useLanguageFromSettings();
  useRuleAlerts();
  return (
    <TooltipProvider>
      <Shell>
        <Suspense
          fallback={
            <div className="flex h-full items-center justify-center">
              <Spinner />
            </div>
          }
        >
          <Outlet />
        </Suspense>
      </Shell>
      <Toaster
        position="bottom-right"
        offset={{ bottom: 72, right: 16 }}
        closeButton
        toastOptions={{ className: "font-sans" }}
      />
    </TooltipProvider>
  );
}

export const router = createBrowserRouter([
  {
    element: <Layout />,
    children: [
      { path: "/", element: <InspectPage /> },
      { path: "/profiles", element: <ProfilesPage /> },
      { path: "/profiles/:id", element: <ProfilesPage /> },
      { path: "/reports", element: <ReportsPage /> },
      { path: "/integrations/*", element: <IntegrationsPage /> },
      { path: "/debug", element: <DebugPage /> },
      { path: "/system", element: <SystemPage /> },
      { path: "*", element: <InspectPage /> },
    ],
  },
]);
