// Model tab (read-only): validation results of the CLI run for all 7 classes (§15.5).
import { Info } from "lucide-react";
import { useTranslation } from "react-i18next";

import { ClassLabel } from "@/components/status";
import { Spinner } from "@/components/ui/primitives";
import { fmtDate, fmtMs, fmtNumber, fmtPercent } from "@/lib/format";
import { useModel } from "@/lib/queries";
import { CLASS_NAMES } from "@/lib/types";

export function ConfusionMatrix({ matrix, caption }: { matrix: number[][]; caption: string }) {
  const { t } = useTranslation();
  const max = Math.max(1, ...matrix.flat());
  return (
    <div className="overflow-x-auto">
      <table className="border-collapse text-[12px]">
        <caption className="mb-1 text-left text-[12px] text-muted">{caption}</caption>
        <thead>
          <tr>
            <th className="px-1 text-left text-[11px] font-medium text-muted">
              {t("model.trueLabel")} ↓ / {t("model.predictedLabel")} →
            </th>
            {CLASS_NAMES.map((c, i) => (
              <th
                key={c}
                scope="col"
                className="px-1 pb-1 font-mono text-[11px] font-medium text-ink-2"
                title={c}
              >
                {i}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={CLASS_NAMES[i]}>
              <th scope="row" className="pr-2 text-left font-normal">
                <span className="font-mono text-[11px] text-muted">{i}</span>{" "}
                <ClassLabel name={CLASS_NAMES[i]} />
              </th>
              {row.map((value, j) => {
                const alpha = value / max;
                const diagonal = i === j;
                return (
                  <td
                    key={j}
                    className="h-8 w-10 border border-panel text-center font-mono"
                    style={{
                      background: value
                        ? diagonal
                          ? `rgba(30,123,52,${0.15 + alpha * 0.7})`
                          : `rgba(198,40,40,${0.25 + alpha * 0.6})`
                        : "#F3F4F6",
                      color: value && alpha > 0.5 ? "white" : "#1B1F24",
                    }}
                  >
                    {value}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function ModelTab({ modelRef }: { modelRef: string }) {
  const { t } = useTranslation();
  const { data: model, isLoading, error } = useModel(modelRef);
  if (isLoading) return <Spinner />;
  if (error || !model) return <p className="text-reject">{t("model.notFound")}</p>;
  const metrics = model.metrics;
  const validation = metrics.validation;
  return (
    <div className="flex flex-col gap-4">
      <p className="flex items-start gap-2 rounded-[6px] bg-info-tint px-3 py-2 text-[13px] text-info">
        <Info className="mt-0.5 size-4 shrink-0" aria-hidden="true" />
        <span>
          {t("model.trainedByCli")} <code className="font-mono">make run</code>.{" "}
          {t("model.notHere")}
        </span>
      </p>
      {model.version_warning ? (
        <p className="rounded-[6px] bg-review-tint px-3 py-2 text-[13px] text-review">
          {t("model.versionWarning")}
        </p>
      ) : null}
      <dl className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {[
          [
            t("model.featureVector"),
            <span className="font-mono">{model.feature_vector ?? "-"}</span>,
          ],
          [t("model.algorithm"), <span className="font-mono">{model.algorithm}</span>],
          [
            t("model.accuracy"),
            <span className="font-mono text-[20px] font-semibold">
              {fmtPercent(validation?.accuracy, 1)}
            </span>,
          ],
          [
            t("model.f1Macro"),
            <span className="font-mono text-[20px] font-semibold">
              {fmtNumber(validation?.f1_macro, 3)}
            </span>,
          ],
          [
            t("model.cvScore"),
            <span className="font-mono">{fmtNumber(metrics.cv_f1_macro, 4)}</span>,
          ],
          [t("model.trainedAt"), fmtDate(model.trained_at)],
          [t("model.featureLength"), <span className="font-mono">{model.feature_length}</span>],
          [
            t("model.extractTime"),
            <span className="font-mono">{fmtMs(metrics.extract_ms_per_image)}</span>,
          ],
          [
            t("model.predictTime"),
            <span className="font-mono">{fmtMs(metrics.predict_ms_per_image)}</span>,
          ],
        ].map(([label, value], i) => (
          <div key={i} className="rounded-[6px] bg-sunken px-3 py-2">
            <dt className="text-[12px] text-ink-2">{label}</dt>
            <dd className="mt-0.5">{value}</dd>
          </div>
        ))}
      </dl>
      <section>
        <h4 className="mb-1.5 font-semibold">{t("model.bestParams")}</h4>
        <p className="font-mono text-[12px]">{JSON.stringify(model.best_params)}</p>
      </section>
      <section>
        <h4 className="mb-1.5 font-semibold">{t("model.perClass")}</h4>
        <table className="w-full text-[13px]">
          <thead>
            <tr className="border-b border-line text-left text-[12px] text-ink-2">
              <th className="py-1.5 font-medium">{t("model.class")}</th>
              <th className="py-1.5 text-right font-medium">{t("model.precision")}</th>
              <th className="py-1.5 text-right font-medium">{t("model.recall")}</th>
              <th className="py-1.5 text-right font-medium">F1</th>
              <th className="py-1.5 text-right font-medium">{t("model.support")}</th>
            </tr>
          </thead>
          <tbody>
            {CLASS_NAMES.map((name) => {
              const row = metrics.per_class?.[name] ?? {};
              return (
                <tr key={name} className="border-b border-line last:border-0">
                  <td className="py-1">
                    <ClassLabel name={name} withDescription />
                  </td>
                  <td className="py-1 text-right font-mono">{fmtNumber(row.precision, 4)}</td>
                  <td className="py-1 text-right font-mono">{fmtNumber(row.recall, 4)}</td>
                  <td className="py-1 text-right font-mono">{fmtNumber(row.f1, 4)}</td>
                  <td className="py-1 text-right font-mono">{row.support ?? "—"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </section>
      {metrics.confusion_matrix ? (
        <section>
          <h4 className="mb-1.5 font-semibold">{t("model.confusion")}</h4>
          <ConfusionMatrix
            matrix={metrics.confusion_matrix}
            caption={t("model.confusionCaption")}
          />
        </section>
      ) : null}
      <section className="grid gap-4 sm:grid-cols-2">
        <div>
          <h4 className="mb-1.5 font-semibold">{t("model.trainingParams")}</h4>
          <dl className="max-h-72 overflow-auto rounded-[6px] bg-sunken p-2 font-mono text-[11px]">
            {Object.entries(model.training_params).map(([key, value]) => (
              <div key={key} className="flex justify-between gap-3">
                <dt className="text-ink-2">{key}</dt>
                <dd className="text-right">{JSON.stringify(value)}</dd>
              </div>
            ))}
          </dl>
        </div>
        <div>
          <h4 className="mb-1.5 font-semibold">{t("model.libraries")}</h4>
          <dl className="rounded-[6px] bg-sunken p-2 font-mono text-[12px]">
            {Object.entries(model.library_versions).length ? (
              Object.entries(model.library_versions).map(([key, value]) => (
                <div key={key} className="flex justify-between gap-3">
                  <dt className="text-ink-2">{key}</dt>
                  <dd>{value}</dd>
                </div>
              ))
            ) : (
              <p className="text-muted">{t("model.noLibraries")}</p>
            )}
          </dl>
          {Object.entries(model.figures).length ? (
            <div className="mt-3 flex flex-col gap-2">
              {Object.entries(model.figures).map(([name, url]) => (
                <a
                  key={name}
                  href={url}
                  target="_blank"
                  rel="noreferrer"
                  className="text-[13px] text-accent underline"
                >
                  {name}
                </a>
              ))}
            </div>
          ) : null}
        </div>
      </section>
    </div>
  );
}
