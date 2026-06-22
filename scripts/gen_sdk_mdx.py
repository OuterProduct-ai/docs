#!/usr/bin/env python3
"""Generate Mintlify MDX for the OuterProduct SDK from its docstrings.

Assumes docstrings follow DOCSTRINGS.md (Google-style, Markdown body, types from
annotations), so there is NO rST conversion — griffe gives structured sections and
we paste them into Mintlify components. One page per concept; each class renders
its methods/properties nested underneath. The proprietary `reasoning` surface is
excluded; a leak-guard reports any leftover mention.

Local source:
  OP_SRC_PATHS="<sdk>/src:<http-types>/src:<training>/src:<dataframe>/python" \
  uv run --python 3.12 --with griffe --with griffe-pydantic python scripts/gen_sdk_mdx.py
PyPI:
  uv run --python 3.12 --with outerproduct --with griffe --with griffe-pydantic \
    python scripts/gen_sdk_mdx.py
"""
from __future__ import annotations

import os
import re
import sys

import griffe

PKG = "outerproduct"
SEARCH_PATHS = [p for p in os.environ.get("OP_SRC_PATHS", "").split(":") if p]

# Not documented: package version + the `Task` union alias (not a class).
# Reasoning IS documented — ReasoningModel + op.reasoning.fit are featured.
EXCLUDE = {"__version__", "Task"}
SKIP_ATTRS = {"model_config", "model_fields", "model_computed_fields",
              "model_extra", "model_fields_set"}
# Inherited framework (Context) methods that are part of the SDK contract.
INHERITED_ALLOW = {"table", "sql", "register_s3", "list_tables"}

PAGES = [
    {"slug": "sdk/overview", "title": "outerproduct", "sidebar": "Overview",
     "desc": "Setup and top-level helpers.",
     "symbols": ["init", "use_client", "use_aclient", "get_model", "get_models", "col", "lit"]},
    {"slug": "sdk/workspace", "title": "Workspace", "sidebar": "Workspace",
     "desc": "The SDK's single data-entry object.", "symbols": ["Workspace"]},
    {"slug": "sdk/model", "title": "Models", "sidebar": "Models",
     "desc": "The two model types: trainer models and reasoning models.",
     "symbols": ["Model", "ReasoningModel"]},
    {"slug": "sdk/reasoning", "title": "Reasoning", "sidebar": "Reasoning",
     "desc": "Train reasoning models and read their explanations and counterfactuals.",
     "symbols": ["reasoning.fit", "Reasoning", "ScenarioResult", "QueryResult", "Counterfactual", "Change"]},
    {"slug": "sdk/trainer", "title": "Trainer", "sidebar": "Trainer",
     "desc": "Configure and run training.", "symbols": ["Trainer", "Metric"]},
    {"slug": "sdk/jobs", "title": "Jobs", "sidebar": "Jobs",
     "desc": "Async job handles.",
     "symbols": ["Job", "TrainingJob", "ReasoningFitJob", "JobResult", "JobFailed", "JobNotReady"]},
    {"slug": "sdk/tasks", "title": "Tasks", "sidebar": "Tasks",
     "desc": "Supervised-learning task configs.",
     "symbols": ["TaskKind", "Regression", "Binclass", "Multiclass", "Forecasting",
                 "SequenceRegression", "SequenceBinclass", "SequenceMulticlass"]},
    {"slug": "sdk/hpo", "title": "Hyperparameter search", "sidebar": "HPO search",
     "desc": "Search space and optimizer.",
     "symbols": ["HPOSpace", "ModelParamSpace", "Float", "Int", "Categorical", "Mixture", "Optimizer"]},
]


# --- griffe helpers ------------------------------------------------------------

def clean_type(s) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    if len(s) >= 2 and s[0] in "\"'" and s[-1] == s[0]:   # forward-ref quotes
        s = s[1:-1]
    return s.replace("builtins.", "").strip()


def esc(s: str) -> str:
    return s.replace('"', "'")


def resolve(o):
    if getattr(o, "is_alias", False):
        try:
            return o.final_target
        except Exception:
            return None
    return o


def parsed(o):
    d = getattr(o, "docstring", None)
    return d.parsed if d else []


def kindof(sec) -> str:
    return getattr(getattr(sec, "kind", None), "value", "")


def has_doc(o) -> bool:
    d = getattr(o, "docstring", None)
    return bool(d and (d.value or "").strip())


def is_prop(o) -> bool:
    return getattr(o, "is_attribute", False) and "property" in (getattr(o, "labels", set()) or set())


def get_default(p):
    for a in ("default", "value"):
        v = getattr(p, a, None)
        if v is not None:
            return str(v)
    return None


def body_text(o) -> str:
    """Summary + description + examples, pasted as Markdown (no conversion)."""
    out = []
    for sec in parsed(o):
        k = kindof(sec)
        if k == "text":
            out.append(sec.value)
        elif k == "examples":
            v = sec.value
            out.append(v if isinstance(v, str)
                       else "\n".join(p[1] if isinstance(p, tuple) else str(p) for p in v))
    return "\n\n".join(c for c in out if c).strip()


# --- rendering -----------------------------------------------------------------

def signature(fn) -> str:
    bits, star = [], False
    for p in getattr(fn, "parameters", []) or []:
        k = getattr(getattr(p, "kind", None), "name", "")
        if p.name in ("self", "cls"):
            continue
        if k == "keyword_only" and not star:
            bits.append("*"); star = True
        if k == "var_positional":
            star = True; bits.append("*" + p.name); continue
        if k == "var_keyword":
            bits.append("**" + p.name); continue
        s = p.name
        if getattr(p, "annotation", None) is not None:
            s += f": {clean_type(p.annotation)}"
        d = get_default(p)
        if d is not None:
            s += f" = {d}"
        bits.append(s)
    ret = getattr(fn, "returns", None)
    return f"{fn.name}({', '.join(bits)})" + (f" -> {clean_type(ret)}" if ret is not None else "")


def param_fields(fn) -> list[str]:
    sig = {p.name: p for p in (getattr(fn, "parameters", []) or [])}
    out = []
    for sec in parsed(fn):
        if kindof(sec) != "parameters":
            continue
        for p in sec.value:
            sp = sig.get(p.name)
            typ = clean_type(getattr(sp, "annotation", None) if sp else None)
            req = "" if (sp and get_default(sp) is not None) else " required"
            t = f' type="{esc(typ)}"' if typ else ""
            out.append(f'<ParamField body="{p.name}"{t}{req}>\n  {(p.description or "").strip()}\n</ParamField>')
    return out


def response_field(fn) -> list[str]:
    secs = [s for s in parsed(fn) if kindof(s) == "returns"]
    if not secs:
        return []
    typ = clean_type(getattr(fn, "returns", None))
    # google splits a wrapped Returns description into one entry per line — rejoin.
    desc = " ".join((getattr(r, "description", "") or "").strip()
                    for sec in secs for r in sec.value).strip()
    t = f' type="{esc(typ)}"' if typ else ""
    return [f'<ResponseField name="returns"{t}>\n  {desc}\n</ResponseField>']


def attr_fields(cls) -> list[str]:
    """Fields documented in the class docstring's Google `Attributes:` section
    (dataclasses document fields there, not on the members themselves)."""
    out = []
    for sec in parsed(cls):
        if kindof(sec) != "attributes":
            continue
        for a in sec.value:
            typ = clean_type(getattr(a, "annotation", None))
            t = f' type="{esc(typ)}"' if typ else ""
            out.append(f'<ParamField body="{a.name}"{t}>\n  {(getattr(a, "description", "") or "").strip()}\n</ParamField>')
    return out


def field_block(o, comp) -> str:
    typ = clean_type(getattr(o, "annotation", None))
    t = f' type="{esc(typ)}"' if typ else ""
    attr = "name" if comp == "ResponseField" else "body"
    return f'<{comp} {attr}="{o.name}"{t}>\n  {body_text(o)}\n</{comp}>'


def render_callable(o, title, level="##") -> str:
    sub = "### " if level == "##" else "**"
    end = "" if level == "##" else "**"
    parts = [f"{level} `{title}`\n", f"```python\n{signature(o)}\n```\n"]
    body = body_text(o)
    if body:
        parts.append(body + "\n")
    pf = param_fields(o)
    if pf:
        parts.append(f"{sub}Parameters{end}\n\n" + "\n\n".join(pf) + "\n")
    rf = response_field(o)
    if rf:
        parts.append(f"{sub}Returns{end}\n\n" + "\n\n".join(rf) + "\n")
    return "\n".join(parts)


def class_members(cls):
    m = dict(cls.members or {})
    try:
        for nm, x in (cls.inherited_members or {}).items():
            if nm in INHERITED_ALLOW:
                m.setdefault(nm, x)
    except Exception:
        pass
    return m


def render_class(cls) -> str:
    parts = [f"## `{cls.name}`\n"]
    body = body_text(cls)
    if body:
        parts.append(body + "\n")
    props, methods, attrs = [], [], []
    for nm, m in class_members(cls).items():
        r = resolve(m)
        if r is None or nm.startswith("_") or nm in EXCLUDE or nm in SKIP_ATTRS:
            continue
        if is_prop(r):
            if has_doc(r):
                props.append(r)
        elif getattr(r, "is_function", False):
            methods.append(r)
        elif getattr(r, "is_attribute", False) and has_doc(r):
            attrs.append(r)
    # Fields: prefer the docstring `Attributes:` section (dataclasses), else
    # documented attribute members (e.g. pydantic Field(description=...)).
    fields = attr_fields(cls) or [field_block(a, "ParamField") for a in attrs]
    if fields:
        parts.append("### Fields\n\n" + "\n\n".join(fields) + "\n")
    if props:
        parts.append("### Properties\n\n" + "\n\n".join(field_block(p, "ResponseField") for p in props) + "\n")
    for mth in methods:
        parts.append(render_callable(mth, f"{cls.name}.{mth.name}()", "###"))
    return "\n".join(parts)


def render_symbol(pkg, name):
    o = pkg
    for part in name.split("."):          # supports dotted paths e.g. reasoning.fit
        o = resolve((o.members or {}).get(part)) if o is not None else None
    if o is None:
        return None
    if getattr(o, "is_class", False):
        return render_class(o)
    if getattr(o, "is_function", False):
        return render_callable(o, f"op.{name}()")
    return None


# --- cross-references ----------------------------------------------------------

XREF = re.compile(r"\[([^\]]+)\]\[([\w.]+)\]")   # [display][canonical.path]


def slug(text: str) -> str:
    """Replicate Mintlify's heading-anchor slug (verified against rendered ids)."""
    s = text.lower().replace("(", "").replace(")", "").replace(".", "-").replace(" ", "-")
    s = re.sub(r"[^a-z0-9_-]", "", s)
    return re.sub(r"-{2,}", "-", s).strip("-")


def index_headings(url, content, index):
    """Map each rendered heading to its page#anchor, keyed by short name."""
    for line in content.splitlines():
        m = re.match(r"#{2,3} `(.+?)`\s*$", line)
        if not m:
            continue
        text = m.group(1)
        bare = text.replace("()", "")
        key = bare[3:].split(".")[-1] if bare.startswith("op.") else bare  # op.reasoning.fit() -> fit
        index.setdefault(key, f"{url}#{slug(text)}")


def resolve_xrefs(content, index):
    linked, dead = [], []

    def repl(m):
        disp, path = m.group(1), m.group(2)
        parts = path.split(".")
        url = index.get(".".join(parts[-2:])) or index.get(parts[-1])
        if url:
            linked.append(path)
            return f"[{disp}]({url})"
        dead.append(path)            # unknown/excluded target -> plain text
        return disp

    return XREF.sub(repl, content), linked, dead


# --- main ---------------------------------------------------------------------

def load():
    try:
        ext = griffe.load_extensions("griffe_pydantic")
    except Exception as e:
        ext = None
        print(f"  warn: griffe_pydantic not loaded ({e})", file=sys.stderr)
    loader = griffe.GriffeLoader(docstring_parser="google", extensions=ext,
                                 search_paths=SEARCH_PATHS or None)
    pkg = loader.load(PKG)
    for extra in ("outerproduct_http_types", "outerproduct_dataframe", "outerproduct_training"):
        try:
            loader.load(extra)
        except Exception:
            pass
    try:
        loader.resolve_aliases(implicit=False, external=True)
    except Exception:
        pass
    return pkg


def main() -> int:
    pkg = load()
    rendered, index = [], {}
    for page in PAGES:                       # pass 1: render + index every heading
        blocks = [render_symbol(pkg, n) for n in page["symbols"] if n not in EXCLUDE]
        blocks = [b for b in blocks if b]
        front = (f'---\ntitle: "{page["title"]}"\nsidebarTitle: "{page["sidebar"]}"\n'
                 f'description: "{esc(page["desc"])}"\n---\n\n'
                 "{/* AUTO-GENERATED by scripts/gen_sdk_mdx.py — do not edit. */}\n\n")
        content = front + "\n".join(blocks).rstrip() + "\n"
        index_headings("/" + page["slug"], content, index)
        rendered.append((page, content, len(blocks)))
    linked, dead = 0, []
    for page, content, n in rendered:        # pass 2: resolve cross-refs (needs full index)
        content, r, u = resolve_xrefs(content, index)
        linked += len(r); dead += u
        with open(page["slug"] + ".mdx", "w") as fh:
            fh.write(content)
        print(f"  {page['slug']+'.mdx':22} {n} symbols")
    print(f"\ncross-refs: {linked} linked, {len(dead)} unresolved")
    for path in sorted(set(dead)):
        print(f"  ✗ unresolved xref: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
