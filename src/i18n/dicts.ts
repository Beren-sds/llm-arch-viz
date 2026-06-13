/**
 * The real EN/ZH dictionaries, bundled for the app. Tests build their own
 * dicts; this is the single place the shipped narration JSON is wired into
 * an I18n instance (main.ts / pages).
 */

import type { Dicts } from "./i18n";
import en from "./en.json";
import zh from "./zh.json";

export const DICTS: Dicts = { en, zh };
