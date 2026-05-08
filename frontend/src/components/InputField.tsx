type InputFieldProps = {
  label: string;
  name: string;
  value: string | number;
  type?: "text" | "number";
  min?: number;
  max?: number;
  step?: number;
  helper?: string;
  onChange: (name: string, value: string) => void;
};

export default function InputField({
  label,
  name,
  value,
  type = "number",
  min,
  max,
  step,
  helper,
  onChange,
}: InputFieldProps) {
  return (
    <label className="block">
      <span className="mb-2 block text-sm font-medium text-graphite">{label}</span>
      <input
        className="h-11 w-full rounded-md border border-line bg-white px-3 text-sm text-ink outline-none transition focus:border-signal focus:ring-2 focus:ring-signal/20"
        name={name}
        value={value}
        type={type}
        min={min}
        max={max}
        step={step}
        onChange={(event) => onChange(name, event.target.value)}
      />
      {helper ? <span className="mt-1 block text-xs text-slate-500">{helper}</span> : null}
    </label>
  );
}
