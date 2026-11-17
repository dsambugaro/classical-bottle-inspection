// Types of the backend API (bottle_inspection_web). Field names follow the API (snake_case).

export const CLASS_NAMES = [
  "bottle_no_defect",
  "bottle_other_brand",
  "defect_low_fill",
  "defect_no_cap",
  "defect_loose_cap",
  "defect_debris",
  "defect_label_damage",
] as const;
export type ClassName = (typeof CLASS_NAMES)[number];

export const CLASS_COLORS: Record<ClassName, string> = {
  bottle_no_defect: "#2B8A3E",
  bottle_other_brand: "#868E96",
  defect_low_fill: "#1C7ED6",
  defect_no_cap: "#E03131",
  defect_loose_cap: "#F08C00",
  defect_debris: "#7048E8",
  defect_label_damage: "#C2255C",
};

export type Verdict = "pass" | "reject" | "review";
export type ProfileStatusName = "ready" | "modified" | "model_missing" | "incompatible";
export type Rect = [number, number, number, number];

export interface ApiErrorBody {
  error: { code: string; message: string; details: Record<string, unknown> };
}

export interface ModelSummary {
  model_ref: string;
  folder: string;
  /** Feature vector the model was trained on (e.g. "rois_hog"). */
  feature_vector: string | null;
  algorithm: string;
  display_name: string;
  status: "ok" | "missing" | "error";
  file_mtime: number;
  trained_at: string;
  updated_on: string | null;
  discovered_at: string | null;
  accuracy: number | null;
  f1_macro: number | null;
  feature_length: number;
  version_warning: string | null;
  load_error: string | null;
  missing: boolean;
}

export interface PerClassMetrics {
  precision?: number;
  recall?: number;
  f1?: number;
  support?: number;
}

export interface ModelDetail extends ModelSummary {
  training_params: Record<string, unknown>;
  best_params: Record<string, unknown>;
  library_versions: Record<string, string>;
  classes: ClassName[];
  figures: Record<string, string>;
  metrics: {
    display_name?: string;
    validation?: { accuracy: number; f1_macro: number; f1_per_class: Record<string, number> };
    cv_f1_macro?: number;
    per_class: Record<ClassName, PerClassMetrics>;
    confusion_matrix?: number[][] | null;
    comparison?: Record<string, unknown>;
    predict_ms_per_image?: number;
    extract_ms_per_image?: number;
    train_seconds?: number;
    facts?: Record<string, number>;
    photometric_normalization?: boolean;
  };
}

export interface ProfileStatusDetail {
  status: ProfileStatusName;
  can_analyse: boolean;
  modified_keys: string[];
  locked_mismatch: string[];
  reason: string | null;
  model_updated_on: string | null;
}

export interface VerdictSettings {
  reject_classes: ClassName[];
  min_confidence: number | null;
  low_confidence_verdict: "reject" | "review";
}

export type ParamValue = number | boolean | number[] | number[][] | string[] | string | null;
export type Params = Record<string, ParamValue>;

export interface Profile {
  id: number;
  name: string;
  description: string;
  color: string;
  model_ref: string;
  algorithm: string;
  background_source: "model" | "captured" | "uploaded";
  has_background: boolean;
  version: number;
  status: ProfileStatusName;
  status_detail: ProfileStatusDetail;
  is_builtin: boolean;
  deleted: boolean;
  created_at: string;
  updated_at: string;
  model: ModelSummary | null;
  params: Params;
  verdict: VerdictSettings;
  training_params?: Params | null;
}

export interface CatalogEntry {
  key: string;
  group: string;
  type:
    | "int"
    | "float"
    | "bool"
    | "range"
    | "ranges"
    | "rect"
    | "size"
    | "int3"
    | "enum"
    | "multiselect";
  mode: "editable" | "locked";
  config_constant: string;
  min: number | null;
  max: number | null;
  step: number | null;
  unit: string | null;
  affects_preview: boolean;
  options: string[] | null;
  constraints: string[];
  nullable: boolean;
  label_key: string;
  help_key: string;
  default: ParamValue;
}

export interface Catalog {
  groups: string[];
  entries: CatalogEntry[];
}

export interface Geometry {
  bottle_center_x: number;
  bottle_top_y: number;
  bottle_box: Rect;
  /** Regions of the model's feature groups (cap/label/body, auto_1..3), in image coordinates. */
  rois: Partial<Record<"cap" | "label" | "body" | "auto_1" | "auto_2" | "auto_3", Rect>>;
  reference_patch: Rect;
  foreground_fraction: number;
}

export interface Timings {
  decode: number;
  normalize: number;
  foreground_mask: number;
  localize: number;
  crop_regions: number;
  features: Record<string, number>;
  predict: number;
  total: number;
}

export interface ImageRef {
  id: number | null;
  width: number | null;
  height: number | null;
  url: string | null;
  thumb_url: string | null;
}

export interface KnnNeighbor {
  class: ClassName;
  distance: number;
  train_index: number;
  image?: string;
}

export interface AnalysisResult {
  analysis_id: number;
  inspection_id: number;
  profile: {
    id: number;
    name: string;
    version: number;
    algorithm: string;
    model_ref: string;
    status: "ready" | "modified";
    color: string;
  };
  image: ImageRef;
  predicted_class: ClassName;
  verdict: Verdict;
  confidence: number;
  scores: Record<ClassName, number | null>;
  score_kind: "svm_ovr_score" | "knn_class_distance" | "rf_probability" | string;
  details: {
    svm_votes?: Record<ClassName, number>;
    neighbors?: KnnNeighbor[];
    k?: number;
    n_trees?: number;
    lower_is_better?: boolean;
  };
  geometry: Geometry;
  timings_ms: Timings;
  ground_truth: ClassName | null;
  warnings: string[];
  source: string;
  session_id: number | null;
  created_at: string;
}

export interface InspectionResult {
  inspection_id: number;
  session_id: number | null;
  source: string;
  original_name: string;
  ground_truth: ClassName | null;
  image: ImageRef | null;
  created_at: string;
  analyses: AnalysisResult[];
}

/** A row of GET /api/history: an inspection with all its analyses. */
export interface HistoryItem extends InspectionResult {
  camera_name: string | null;
}

export interface HistoryPage {
  items: HistoryItem[];
  /** Pass as `before` to get the next page; null on the last one. */
  next_cursor: number | null;
  has_more: boolean;
  limit: number;
}

export interface DatasetSample {
  name: string;
  split: string;
  ref: string;
  label: ClassName;
  url: string;
  thumb_url: string;
}

export interface PreviewResult {
  geometry: Geometry;
  image: { width: number; height: number };
  modified_keys: string[];
  ground_truth: ClassName | null;
  masks?: { foreground: string; size: [number, number] };
  normalized_image?: string;
  classification?: {
    predicted_class: ClassName;
    verdict: Verdict;
    confidence: number;
    scores: Record<ClassName, number | null>;
    score_kind: string;
    total_ms: number;
  };
}

export interface AppNotification {
  id: number;
  level: "info" | "warning" | "critical" | "error";
  title: string;
  message: string;
  code: string | null;
  params: Record<string, unknown>;
  link: string | null;
  read_at: string | null;
  created_at: string;
}

export type Settings = {
  language: "en" | "pt-BR";
  storage_policy: "all" | "rejects_only" | "none";
  image_retention_days: number;
  execution_retention_days: number;
  model_folders: string[];
  dataset_path: string;
  max_upload_mb: number;
  max_batch_files: number;
  max_profiles: number;
  default_fps: number;
  min_fps: number;
  max_fps: number;
  background_capture_frames: number;
  motion_threshold: number;
};

// ------------------------------------------------------------------ debug mode

export interface DebugSession {
  id: string;
  name: string;
  width: number;
  height: number;
  ground_truth: ClassName | null;
  image_url: string;
}

/** BGR values of a small square of pixels: values[row][column] = [b, g, r]. */
export interface PixelPatch {
  x: number;
  y: number;
  values: number[][][];
}

export interface DebugRegion {
  name: string;
  kind: "adhoc" | "auto" | "bottle";
  band: [number, number];
  rect: Rect;
  shape: number[];
  canonical_size: [number, number];
  image: string;
}

/** Every intermediate result of the preprocessing (images are data URIs). */
export interface PreprocessingTrace {
  profile_id: number;
  key: string;
  params: Params;
  original: { shape: number[]; dtype: string; patch: PixelPatch };
  background: { image: string; shape: number[] };
  normalization: {
    enabled: boolean;
    reference_patch: Rect;
    patch_mean_background: number[];
    patch_mean_image: number[];
    gain: number[];
    image: string;
    patch: PixelPatch;
  };
  difference: {
    size: [number, number];
    small: string;
    heatmap: string;
    max: number;
    clip: number;
    histogram: number[];
    otsu: number;
    noise_floor: number;
    threshold: number;
  };
  mask: {
    raw: string;
    closed: string;
    opened: string;
    close_size: number;
    open_size: number;
    fraction_raw: number;
    fraction_closed: number;
    fraction: number;
  };
  localization: {
    columns_raw: number[];
    columns: number[];
    column_threshold: number;
    column_peak: number;
    rows: number[];
    top_threshold: number;
    row_threshold: number;
    x: [number, number];
    centre: number;
    box_x: [number, number];
    bottle_width: number;
    y: [number, number];
    size: [number, number];
    smoothing: number;
    column_fraction: number;
    top_fraction: number;
    row_fraction: number;
    box: Rect;
    box_px: Rect;
    empty: boolean;
  };
  regions: DebugRegion[];
}

export interface VectorStats {
  min: number;
  max: number;
  mean: number;
  std: number;
}

export interface GroupTrace {
  name: string;
  region: string;
  descriptor: "hog" | "lbp" | "color";
  offset: number;
  length: number;
  stats: VectorStats;
  values: number[];
  crop: string;
  crop_shape: number[];
  canonical?: string;
  canonical_size?: [number, number];
  hog?: {
    cell: [number, number];
    cells: [number, number];
    blocks: [number, number];
    block: [number, number];
    orientations: number;
    gradient: string;
    rendering: string;
  };
  lbp?: {
    cell: [number, number];
    cells: [number, number];
    points: number;
    radius: number;
    n_codes: number;
    codes: string;
    palette: string[];
    histogram: number[];
  };
  color?: { bins: [number, number, number]; h: string; s: string; v: string };
}

/** Feature groups, feature vector and classifier decision of one profile on the debug image. */
export interface ProfileTrace {
  preprocessing_key: string;
  feature_vector: string | null;
  groups: GroupTrace[];
  vector: { length: number; stats: VectorStats };
  scaled: {
    values: number[];
    mean_abs: number;
    max_abs: number;
    outliers: number;
    outlier_z: number;
  } | null;
  analysis: AnalysisResult;
}
