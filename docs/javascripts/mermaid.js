document$.subscribe(() => {
  if (typeof mermaid !== "undefined") {
    mermaid.initialize({
      startOnLoad: false,
      theme: "default",
      securityLevel: "loose"
    });
    mermaid.run({
      nodes: document.querySelectorAll("pre.mermaid, div.mermaid, .mermaid code")
    });
  }
});
