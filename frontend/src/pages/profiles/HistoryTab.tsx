// History tab: versions with the diff from the previous one and "restore this version" (REQ-PR4).
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";

import { Button, Spinner } from "@/components/ui/primitives";
import { api } from "@/lib/api";
import { fmtDate } from "@/lib/format";
import { keys } from "@/lib/queries";

interface Change {
  key: string;
  from: unknown;
  to: unknown;
}

interface VersionRow {
  version: number;
  created_at: string;
  changes: Change[];
}

export function ChangeList({ changes }: { changes: Change[] }) {
  const { t } = useTranslation();
  if (!changes.length) return <p className="text-[12px] text-muted">{t("history.noChanges")}</p>;
  return (
    <table className="w-full text-[12px]">
      <tbody>
        {changes.map((c) => (
          <tr key={c.key} className="border-b border-line last:border-0">
            <td className="py-1 pr-3 font-mono text-ink-2">{c.key}</td>
            <td className="py-1 pr-3 font-mono text-reject line-through decoration-1">
              {JSON.stringify(c.from)}
            </td>
            <td className="py-1 font-mono text-pass">{JSON.stringify(c.to)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function HistoryTab({
  profileId,
  current,
  readOnly,
}: {
  profileId: number;
  current: number;
  readOnly: boolean;
}) {
  const { t } = useTranslation();
  const client = useQueryClient();
  const versions = useQuery({
    queryKey: ["profile-versions", profileId, current],
    queryFn: () => api.get<VersionRow[]>(`/api/profiles/${profileId}/versions`),
  });
  const restore = useMutation({
    mutationFn: (version: number) =>
      api.post(`/api/profiles/${profileId}/versions/${version}/restore`),
    onSuccess: (_data, version) => {
      toast.success(t("history.restored", { version }));
      void client.invalidateQueries({ queryKey: keys.profiles });
    },
    onError: () => toast.error(t("errors.unexpected")),
  });
  if (versions.isLoading) return <Spinner />;
  return (
    <ol className="flex flex-col gap-3">
      {versions.data?.map((v) => (
        <li key={v.version} className="rounded-[6px] border border-line p-3">
          <div className="mb-2 flex items-center gap-3">
            <span className="font-mono font-semibold">v{v.version}</span>
            <span className="text-[12px] text-muted">{fmtDate(v.created_at, true)}</span>
            {v.version === current ? (
              <span className="rounded-[4px] bg-info-tint px-1.5 text-[12px] font-medium text-info">
                {t("history.current")}
              </span>
            ) : !readOnly ? (
              <Button
                size="sm"
                className="ml-auto"
                disabled={restore.isPending}
                onClick={() => restore.mutate(v.version)}
              >
                {t("history.restore")}
              </Button>
            ) : null}
          </div>
          {v.version === 1 ? (
            <p className="text-[12px] text-muted">{t("history.created")}</p>
          ) : (
            <ChangeList changes={v.changes} />
          )}
        </li>
      ))}
    </ol>
  );
}
