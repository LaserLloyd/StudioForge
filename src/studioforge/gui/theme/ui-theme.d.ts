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
    contrastProfile: 'standard' | 'high' | 'night';
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

  /** Words ui-components.js puts on the page; see locales/en.json. */
  type UIStringKey = 'cancel' | 'ok' | 'confirm' | 'input' | 'message' | 'value' | 'copy' | 'copied' |
    'copyFailed' | 'reset' | 'resetNamed' | 'basic' | 'advanced' | 'setByEnv';

  interface UISettingController {
    /** Re-read the control and show or hide the reset button. */
    update(): void;
    destroy(): void;
  }

  interface UIComponentsApi {
    confirm(options: UIDialogOptions | string): Promise<boolean>;
    alert(options: UIDialogOptions | string): Promise<void>;
    prompt(options: UIPromptOptions | string): Promise<string | null>;
    toast(message: string, options?: { kind?: 'info' | 'success' | 'warning' | 'danger'; timeout?: number }): HTMLElement;
    /** Copy an element's text (a .ui-copy, a code block, an input's value) or a string. Resolves true on success. */
    copy(target: Element | string): Promise<boolean>;
    /** Mark a field invalid with a message (aria-invalid, aria-describedby, announced); an empty message clears it. */
    validate(field: Element, message: string | null | undefined, options?: { focus?: boolean; announce?: boolean }): HTMLElement | null;
    /** Wire a .ui-setting row: reset-to-default, translated badges, env lock. */
    setting(row: Element, options?: { default?: string | boolean; onReset?: (control: HTMLElement, row: Element) => void }): UISettingController;
    /** Say something politely to assistive technology. */
    announce(message: string): void;
    /** Wire copy buttons and data-default settings rows under a root (default: the document). */
    init(root?: ParentNode): void;
    /** A word by key; data-ui-str-<key> on the element or an ancestor wins. {placeholders} are filled from vars. */
    t(key: UIStringKey | (string & {}), element?: Element | null, vars?: Record<string, string | number>): string;
    strings(): Record<string, string>;
    /** Merge translated words over the table. */
    setStrings(map: Partial<Record<UIStringKey, string>> & Record<string, string>): Record<string, string>;
    resetStrings(): Record<string, string>;
    /** Fetch locales/<lang>.json next to ui-components.js and merge it; resolves true when applied. */
    loadLocale(lang?: string, url?: string): Promise<boolean>;
    /** The page language (<html lang>), else the browser's. */
    locale(): string;
    formatNumber(value: number | bigint, options?: Intl.NumberFormatOptions): string;
    formatDate(value: Date | number | string, options?: Intl.DateTimeFormatOptions): string;
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
