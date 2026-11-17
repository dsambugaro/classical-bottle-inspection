// shadcn/ui-style primitives built on Radix, styled with the instrument-panel tokens.
import { cva, type VariantProps } from "class-variance-authority";
import { clsx, type ClassValue } from "clsx";
import { Check, ChevronDown, X } from "lucide-react";
import {
  Checkbox as RCheckbox,
  Dialog as RDialog,
  DropdownMenu as RDropdown,
  Popover as RPopover,
  Select as RSelect,
  Slider as RSlider,
  Switch as RSwitch,
  Tabs as RTabs,
  Tooltip as RTooltip,
} from "radix-ui";
import { forwardRef, type ComponentPropsWithoutRef, type ReactNode } from "react";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

// ------------------------------------------------------------------ Button

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-1.5 whitespace-nowrap rounded-[6px] font-medium transition-colors duration-150 ease-out disabled:pointer-events-none disabled:opacity-50 [&_svg]:size-4 [&_svg]:shrink-0",
  {
    variants: {
      variant: {
        primary: "bg-accent text-white hover:bg-accent-hover",
        secondary: "border border-line bg-panel text-ink hover:bg-sunken",
        ghost: "text-ink-2 hover:bg-sunken hover:text-ink",
        danger: "bg-reject text-white hover:bg-[#a32020]",
        "danger-outline": "border border-reject/40 bg-panel text-reject hover:bg-reject-tint",
      },
      size: {
        sm: "h-7 px-2.5 text-[13px]",
        md: "h-8 px-3 text-[13px]",
        lg: "h-10 px-4 text-sm",
        icon: "size-8",
      },
    },
    defaultVariants: { variant: "secondary", size: "md" },
  },
);

export type ButtonProps = ComponentPropsWithoutRef<"button"> & VariantProps<typeof buttonVariants>;

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { className, variant, size, type = "button", ...props },
  ref,
) {
  return (
    <button
      ref={ref}
      type={type}
      className={cn(buttonVariants({ variant, size }), className)}
      {...props}
    />
  );
});

// ------------------------------------------------------------------ Tooltip

export function Tip({
  content,
  children,
  side = "top",
}: {
  content: ReactNode;
  children: ReactNode;
  side?: "top" | "bottom" | "left" | "right";
}) {
  return (
    <RTooltip.Root delayDuration={250}>
      <RTooltip.Trigger asChild>{children}</RTooltip.Trigger>
      <RTooltip.Portal>
        <RTooltip.Content
          side={side}
          sideOffset={6}
          className="z-50 max-w-80 rounded-[6px] bg-ink px-2.5 py-1.5 text-[12px] leading-snug text-white shadow-[var(--shadow-float)]"
        >
          {content}
          <RTooltip.Arrow className="fill-ink" />
        </RTooltip.Content>
      </RTooltip.Portal>
    </RTooltip.Root>
  );
}

export const TooltipProvider = RTooltip.Provider;

// ------------------------------------------------------------------ Dialog

export function Dialog({
  open,
  onOpenChange,
  title,
  description,
  children,
  footer,
  wide,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: ReactNode;
  description?: ReactNode;
  children?: ReactNode;
  footer?: ReactNode;
  wide?: boolean;
}) {
  return (
    <RDialog.Root open={open} onOpenChange={onOpenChange}>
      <RDialog.Portal>
        <RDialog.Overlay className="fixed inset-0 z-40 bg-ink/30" />
        <RDialog.Content
          className={cn(
            "fixed left-1/2 top-1/2 z-50 max-h-[85vh] w-[calc(100vw-32px)] -translate-x-1/2 -translate-y-1/2 overflow-auto rounded-[10px] bg-panel p-5 shadow-[var(--shadow-float)]",
            wide ? "max-w-3xl" : "max-w-md",
          )}
        >
          <div className="mb-3 flex items-start justify-between gap-4">
            <div>
              <RDialog.Title className="text-base font-semibold">{title}</RDialog.Title>
              {description ? (
                <RDialog.Description className="mt-1 text-[13px] text-ink-2">
                  {description}
                </RDialog.Description>
              ) : (
                <RDialog.Description className="sr-only">{String(title)}</RDialog.Description>
              )}
            </div>
            <RDialog.Close asChild>
              <Button variant="ghost" size="icon" aria-label="Close">
                <X />
              </Button>
            </RDialog.Close>
          </div>
          {children}
          {footer ? <div className="mt-5 flex justify-end gap-2">{footer}</div> : null}
        </RDialog.Content>
      </RDialog.Portal>
    </RDialog.Root>
  );
}

// ------------------------------------------------------------------ Tabs

export const Tabs = RTabs.Root;
export function TabsList({ className, ...props }: ComponentPropsWithoutRef<typeof RTabs.List>) {
  return <RTabs.List className={cn("flex gap-1 border-b border-line", className)} {...props} />;
}
export function TabsTrigger({
  className,
  ...props
}: ComponentPropsWithoutRef<typeof RTabs.Trigger>) {
  return (
    <RTabs.Trigger
      className={cn(
        "-mb-px border-b-2 border-transparent px-3 py-2 text-[13px] font-medium text-ink-2 transition-colors hover:text-ink data-[state=active]:border-accent data-[state=active]:text-ink",
        className,
      )}
      {...props}
    />
  );
}
export const TabsContent = RTabs.Content;

// ------------------------------------------------------------------ Switch / Checkbox

export function Switch({ className, ...props }: ComponentPropsWithoutRef<typeof RSwitch.Root>) {
  return (
    <RSwitch.Root
      className={cn(
        "relative h-5 w-9 shrink-0 rounded-full bg-[#c3c9d2] transition-colors data-[state=checked]:bg-accent disabled:opacity-50",
        className,
      )}
      {...props}
    >
      <RSwitch.Thumb className="block size-4 translate-x-0.5 rounded-full bg-white shadow transition-transform data-[state=checked]:translate-x-[18px]" />
    </RSwitch.Root>
  );
}

export function Checkbox({ className, ...props }: ComponentPropsWithoutRef<typeof RCheckbox.Root>) {
  return (
    <RCheckbox.Root
      className={cn(
        "flex size-4 shrink-0 items-center justify-center rounded-[4px] border border-[#9aa3ae] bg-panel data-[state=checked]:border-accent data-[state=checked]:bg-accent disabled:opacity-50",
        className,
      )}
      {...props}
    >
      <RCheckbox.Indicator>
        <Check className="size-3 text-white" strokeWidth={3} />
      </RCheckbox.Indicator>
    </RCheckbox.Root>
  );
}

// ------------------------------------------------------------------ Select

export function Select({
  value,
  onValueChange,
  options,
  label,
  className,
  disabled,
}: {
  value: string;
  onValueChange: (value: string) => void;
  options: { value: string; label: ReactNode; disabled?: boolean }[];
  label: string;
  className?: string;
  disabled?: boolean;
}) {
  return (
    <RSelect.Root value={value} onValueChange={onValueChange} disabled={disabled}>
      <RSelect.Trigger
        aria-label={label}
        className={cn(
          "inline-flex h-8 min-w-0 items-center justify-between gap-2 rounded-[6px] border border-line bg-panel px-2.5 text-[13px] hover:border-[#b9c0ca] disabled:opacity-50",
          className,
        )}
      >
        <span className="truncate">
          <RSelect.Value />
        </span>
        <RSelect.Icon>
          <ChevronDown className="size-4 text-muted" />
        </RSelect.Icon>
      </RSelect.Trigger>
      <RSelect.Portal>
        <RSelect.Content
          position="popper"
          sideOffset={4}
          className="z-50 max-h-80 min-w-[var(--radix-select-trigger-width)] overflow-auto rounded-[6px] border border-line bg-panel p-1 shadow-[var(--shadow-float)]"
        >
          <RSelect.Viewport>
            {options.map((option) => (
              <RSelect.Item
                key={option.value}
                value={option.value}
                disabled={option.disabled}
                className="relative flex cursor-default select-none items-center rounded-[4px] py-1.5 pl-7 pr-2 text-[13px] outline-none data-[disabled]:opacity-50 data-[highlighted]:bg-sunken"
              >
                <RSelect.ItemIndicator className="absolute left-2">
                  <Check className="size-3.5" />
                </RSelect.ItemIndicator>
                <RSelect.ItemText>{option.label}</RSelect.ItemText>
              </RSelect.Item>
            ))}
          </RSelect.Viewport>
        </RSelect.Content>
      </RSelect.Portal>
    </RSelect.Root>
  );
}

// ------------------------------------------------------------------ Slider

export function Slider({ className, ...props }: ComponentPropsWithoutRef<typeof RSlider.Root>) {
  const count = (props.value ?? props.defaultValue ?? [0]).length;
  return (
    <RSlider.Root
      className={cn("relative flex h-5 w-full touch-none select-none items-center", className)}
      {...props}
    >
      <RSlider.Track className="relative h-1 grow rounded-full bg-[#d0d5dc]">
        <RSlider.Range className="absolute h-full rounded-full bg-accent" />
      </RSlider.Track>
      {Array.from({ length: count }, (_, i) => (
        <RSlider.Thumb
          key={i}
          aria-label={props["aria-label"] ? `${props["aria-label"]} ${i + 1}` : undefined}
          className="relative block size-6 rounded-full after:absolute after:inset-1 after:rounded-full after:border-2 after:border-accent after:bg-panel after:shadow-sm"
        />
      ))}
    </RSlider.Root>
  );
}

// ------------------------------------------------------------------ Popover / Dropdown

export const Popover = RPopover.Root;
export const PopoverTrigger = RPopover.Trigger;
export function PopoverContent({
  className,
  ...props
}: ComponentPropsWithoutRef<typeof RPopover.Content>) {
  return (
    <RPopover.Portal>
      <RPopover.Content
        sideOffset={6}
        align="start"
        className={cn(
          "z-50 rounded-[10px] border border-line bg-panel p-3 shadow-[var(--shadow-float)]",
          className,
        )}
        {...props}
      />
    </RPopover.Portal>
  );
}

export const Dropdown = RDropdown.Root;
export const DropdownTrigger = RDropdown.Trigger;
export function DropdownContent({
  className,
  ...props
}: ComponentPropsWithoutRef<typeof RDropdown.Content>) {
  return (
    <RDropdown.Portal>
      <RDropdown.Content
        sideOffset={4}
        align="end"
        className={cn(
          "z-50 min-w-44 rounded-[6px] border border-line bg-panel p-1 shadow-[var(--shadow-float)]",
          className,
        )}
        {...props}
      />
    </RDropdown.Portal>
  );
}
export function DropdownItem({
  className,
  ...props
}: ComponentPropsWithoutRef<typeof RDropdown.Item>) {
  return (
    <RDropdown.Item
      className={cn(
        "flex cursor-default select-none items-center gap-2 rounded-[4px] px-2 py-1.5 text-[13px] outline-none data-[disabled]:opacity-50 data-[highlighted]:bg-sunken [&_svg]:size-4",
        className,
      )}
      {...props}
    />
  );
}

// ------------------------------------------------------------------ small pieces

export function Kbd({ children }: { children: ReactNode }) {
  return (
    <kbd className="rounded-[4px] border border-line bg-sunken px-1.5 py-px font-mono text-[11px] text-ink-2">
      {children}
    </kbd>
  );
}

export function Field({
  label,
  htmlFor,
  children,
  hint,
}: {
  label: ReactNode;
  htmlFor?: string;
  children: ReactNode;
  hint?: ReactNode;
}) {
  return (
    <div className="flex flex-col gap-1">
      <label htmlFor={htmlFor} className="text-[13px] font-medium text-ink-2">
        {label}
      </label>
      {children}
      {hint ? <p className="text-[12px] text-muted">{hint}</p> : null}
    </div>
  );
}

export function Spinner({ className }: { className?: string }) {
  return (
    <span
      role="status"
      aria-label="Loading"
      className={cn(
        "inline-block size-4 animate-spin rounded-full border-2 border-line border-t-accent",
        className,
      )}
    />
  );
}
