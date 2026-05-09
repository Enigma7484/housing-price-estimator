export type HousingPayload = {
  square_footage: number;
  lot_size: number;
  bedrooms: number;
  bathrooms: number;
  floors: number;
  waterfront: boolean;
  view: number;
  condition: number;
  grade: number;
  year_built: number;
  year_renovated: number;
  zipcode: string;
  latitude?: number | null;
  longitude?: number | null;
  parking: number;
  furnishing_status: "basic" | "standard" | "premium";
  main_road_access: boolean;
  basement: boolean;
  air_conditioning: boolean;
};

export type HousingPrediction = {
  predicted_price: number;
  formatted_price: string;
  model_name: string;
  price_range: {
    low: number;
    high: number;
    formatted_low: string;
    formatted_high: string;
  };
  confidence: string;
  value_badge: string;
  explanation: string[];
  input_summary: Record<string, string | number | boolean>;
};

export type MobilePayload = {
  battery_power: number;
  clock_speed: number;
  ram: number;
  internal_memory: number;
  mobile_weight: number;
  n_cores: number;
  primary_camera_mp: number;
  front_camera_mp: number;
  pixel_height: number;
  pixel_width: number;
  screen_height_cm: number;
  screen_width_cm: number;
  talk_time: number;
  mobile_depth_cm: number;
  bluetooth: boolean;
  dual_sim: boolean;
  four_g: boolean;
  three_g: boolean;
  touch_screen: boolean;
  wifi: boolean;
};

export type MobilePrediction = {
  predicted_price_range: number;
  label: string;
  confidence: number;
  probabilities: Record<string, number>;
  model_name: string;
  explanation: string[];
  input_summary: Record<string, string | number | boolean>;
};

export type CarPayload = {
  make: string;
  model: string;
  body_type: "sedan" | "suv" | "truck" | "hatchback" | "coupe" | "wagon";
  fuel_type: "gasoline" | "diesel" | "hybrid" | "electric";
  transmission: "automatic" | "manual";
  year: number;
  mileage: number;
  engine_size_l: number;
  horsepower: number;
  owners: number;
  accident_history: boolean;
  condition_score: number;
};

export type CarPrediction = {
  predicted_price: number;
  formatted_price: string;
  model_name: string;
  price_range: {
    low: number;
    high: number;
    formatted_low: string;
    formatted_high: string;
  };
  confidence: string;
  value_badge: string;
  explanation: string[];
  input_summary: Record<string, string | number | boolean>;
};

export type ModelMetadata = {
  estimator: string;
  model_name: string;
  metrics: Partial<{
    mae: number;
    rmse: number;
    r2: number;
    accuracy: number;
    f1_macro: number;
  }>;
  trained_at: string;
  dataset_rows: number;
  target: string;
  status: string;
  dataset_source?: string;
};

export type EstimatorCatalogItem = {
  key: string;
  name: string;
  category: string;
  problem_type: "Regression" | "Classification" | string;
  route: string | null;
  frontend_path: string | null;
  phase: string;
  description: string;
  status: "ready" | "planned" | "model_not_loaded" | string;
  metadata: ModelMetadata | null;
};

export type EstimatorCatalogResponse = {
  platform: string;
  live_count: number;
  estimators: EstimatorCatalogItem[];
};
