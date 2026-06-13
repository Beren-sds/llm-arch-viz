/** Tiny DOM construction helpers shared by the page modules. */

export function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  node.className = className;
  return node;
}

export function button(className: string): HTMLButtonElement {
  const b = document.createElement("button");
  b.type = "button";
  b.className = className;
  return b;
}
