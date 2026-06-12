/**
 * Minimal ambient typing for the one Node API the test suite uses.
 *
 * @types/node is deliberately NOT a dependency of this browser-targeted
 * project; loader.test.ts is the only Node-API consumer (it reads the real
 * export artifacts from disk under vitest's node environment). If more Node
 * surface is ever needed, install @types/node instead of growing this shim.
 */
declare module "node:fs" {
  export function readFileSync(path: string | URL, encoding: "utf8"): string;
  export function readFileSync(path: string | URL): Uint8Array<ArrayBuffer>;
}
