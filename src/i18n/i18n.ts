/**
 * Minimal bilingual (EN/ZH) string lookup with locale persistence.
 *
 * Lookup chain for t(key): current locale → EN → the key itself (with a
 * console.error) — a missing translation is always visible on screen and
 * in the console, never a silent empty string.
 *
 * The chosen locale persists to localStorage ('llmviz.locale'). All
 * localStorage access is guarded so the class also works in non-browser
 * environments (vitest node env, SSR); persistence is simply skipped there.
 */

export type Locale = "en" | "zh";

export const LOCALES: readonly Locale[] = ["en", "zh"];

export const LOCALE_STORAGE_KEY = "llmviz.locale";

export type Dicts = Record<Locale, Record<string, string>>;

function isLocale(v: unknown): v is Locale {
  return v === "en" || v === "zh";
}

/** localStorage, or null where it doesn't exist (node, SSR). */
function storage(): Storage | null {
  return typeof localStorage === "undefined" || localStorage === null
    ? null
    : localStorage;
}

/**
 * Keys missing from at least one locale's dictionary, sorted. Empty array
 * means the locales are in sync. Used by ChapterRegistry validation and by
 * the test that keeps the real en.json/zh.json key sets identical.
 */
export function validateDicts(dicts: Dicts): string[] {
  const union = new Set<string>();
  for (const locale of LOCALES) {
    for (const key of Object.keys(dicts[locale])) union.add(key);
  }
  const missing: string[] = [];
  for (const key of union) {
    // Object.hasOwn: keys like 'toString' exist on a plain object's
    // prototype, so an index lookup would wrongly count them as present.
    if (LOCALES.some((locale) => !Object.hasOwn(dicts[locale], key))) {
      missing.push(key);
    }
  }
  return missing.sort();
}

export class I18n {
  private readonly dicts: Dicts;
  private currentLocale: Locale = "en";
  private readonly listeners = new Set<(locale: Locale) => void>();

  constructor(dicts: Dicts) {
    this.dicts = dicts;
    const stored = storage()?.getItem(LOCALE_STORAGE_KEY);
    if (isLocale(stored)) this.currentLocale = stored;
    // anything else (null, garbage) → keep the 'en' default
  }

  get locale(): Locale {
    return this.currentLocale;
  }

  /**
   * Switch locale, persist it, and notify onChange listeners. Setting the
   * locale that is already active is a no-op (no persist, no notify).
   */
  setLocale(locale: Locale): void {
    if (locale === this.currentLocale) return;
    this.currentLocale = locale;
    storage()?.setItem(LOCALE_STORAGE_KEY, locale);
    for (const listener of this.listeners) listener(locale);
  }

  /** Subscribe to locale changes; returns the unsubscribe function. */
  onChange(cb: (locale: Locale) => void): () => void {
    this.listeners.add(cb);
    return () => {
      this.listeners.delete(cb);
    };
  }

  /** Translate `key`: current locale → EN → the key itself (+console.error). */
  t(key: string): string {
    // Object.hasOwn guards against prototype collisions: 'toString' etc.
    // exist on every plain object but are not translations.
    const dict = this.dicts[this.currentLocale];
    if (Object.hasOwn(dict, key)) return dict[key];
    if (Object.hasOwn(this.dicts.en, key)) return this.dicts.en[key];
    console.error(`i18n: missing key "${key}" in every locale`);
    return key;
  }

  /**
   * Locales whose dictionary lacks `key` (empty array = fully translated).
   * Lets callers (ChapterRegistry) validate keys without reaching into the
   * dictionaries.
   */
  missingLocales(key: string): Locale[] {
    return LOCALES.filter((locale) => !Object.hasOwn(this.dicts[locale], key));
  }
}
