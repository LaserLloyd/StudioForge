// TypeScript declarations for the ThemeForge globals.
// Add this file to tsconfig.json's "include" (or reference it with
// /// <reference path="..."/>) to type window.UITheme and window.UIComponents.

export {};

declare global {
  type UIThemeGround = 'oled' | 'dark' | 'light';

  interface UIThemeInfo {
    slug: string;
    name: string;
    family: string;
    ground: UIThemeGround;
    colorScheme: 'dark' | 'light';
    themeColor: string;
    set: 'core' | 'opt-in';
    swatch: string;
    fonts: string[];
  }

  interface UIThemeChange {
    slug: string;
    theme: UIThemeInfo;
    previous: string | null;
    /** true for the temporary switch to a light theme while printing */
    print: boolean;
  }

  interface UIThemeApi {
    readonly version: string;
    readonly config: {
      themes: string[];
      default: string;
      auto: boolean;
      storageKey: string;
      families: boolean;
      printTheme: string;
    };
    current(): string;
    theme(slug?: string): UIThemeInfo | null;
    list(): UIThemeInfo[];
    set(slug: string): string;
    reset(): string;
    partner(slug?: string): string | null;
    toggleFamily(): string | null;
    onChange(listener: (change: UIThemeChange) => void): () => void;
    token(name: string, element?: Element): string;
    tokens<K extends string>(names: K[], element?: Element): Record<K, string>;
    color(name: string): string;
    colors<K extends string>(names: K[]): Record<K, string>;
    mountPicker(
      target: HTMLSelectElement | HTMLElement | string,
      options?: { label?: string; coreLabel?: string; optInLabel?: string; systemLabel?: string }
    ): HTMLSelectElement | null;
  }

  interface UIDialogOptions {
    title?: string;
    message?: string;
    label?: string;
    confirmLabel?: string;
    cancelLabel?: string;
    danger?: boolean;
  }

  interface UIPromptOptions extends UIDialogOptions {
    value?: string;
    placeholder?: string;
    type?: string;
  }

  interface UIComponentsApi {
    confirm(options: UIDialogOptions | string): Promise<boolean>;
    alert(options: UIDialogOptions | string): Promise<void>;
    prompt(options: UIPromptOptions | string): Promise<string | null>;
    toast(message: string, options?: { kind?: 'info' | 'success' | 'warning' | 'danger'; timeout?: number }): HTMLElement;
  }

  interface Window {
    UITheme: UIThemeApi;
    UIComponents: UIComponentsApi;
    UI_THEME_MANIFEST?: Partial<{
      themes: string[];
      default: string;
      defaultDark: string;
      defaultLight: string;
      storageKey: string;
      families: boolean;
      legacy: { key: string; map: Record<string, string> };
      mirrorAttr: string;
      fontsHref: string;
      printTheme: string;
    }>;
  }

  interface DocumentEventMap {
    'ui-theme-change': CustomEvent<UIThemeChange>;
  }
}
