#!/usr/bin/env python3
"""Generate Mintlify MDX for the OuterProduct *SDK* from the published PyPI package.

griffe (static analysis) reads NumPy-style docstrings + type hints + (via the
griffe-pydantic extension) Pydantic field descriptions, and emits one MDX page
per concept. Each class becomes its own section with its methods/properties
nested underneath (so the right-rail "On this page" shows class -> method, like
kumo.ai). All reasoning surface is excluded (proprietary); a leak-guard reports
any leftover "reasoning" mention.

Run:
  uv run --python 3.12 --with "outerproduct==0.6.3" --with griffe \
         --with griffe-pydantic python scripts/gen_sdk_mdx.py
"""
from __future__ import annotations

import json
import re
import sys

import griffe

PKG = "outerproduct"

# Proprietary reasoning surface — never document.
EXCLUDE = {
    "reasoning", "Reasoning", "ScenarioResult", "QueryResult",
    "Counterfactual", "Change", "ReasoningModel", "SequenceReasoningModel",
    "ReasoningFitJob", "__version__", "Task",  # Task is a Union alias, not a class
}

# Pydantic/BaseModel internals we never want to document as fields.
SKIP_ATTRS = {"model_config", "model_fields", "model_computed_fields",
              "model_extra", "model_fields_set"}

# When a class inherits from a framework base (e.g. Workspace <- Context), the
# base drags in low-level plumbing users never call (catalog/schema providers,
# views, deregister, ...). Surface ONLY these *promoted* inherited members.
INHERITED_ALLOW = {"table", "sql", "register_s3", "list_tables"}

# One page per concept; each class within renders as its own section with its
# methods nested (Kumo-style). Free functions render as top-level sections.
PAGES = [
    {"slug": "sdk/overview", "title": "outerproduct", "sidebar": "Overview",
     "desc": "Setup and top-level helpers for the OuterProduct SDK.",
     "intro": "Top-level functions of the `outerproduct` package. Import the SDK as `import outerproduct as op`.",
     "symbols": ["init", "use_client", "use_aclient", "get_model", "get_models", "col", "lit"]},
    {"slug": "sdk/workspace", "title": "Workspace", "sidebar": "Workspace",
     "desc": "The Workspace — the SDK's single data-entry object.",
     "symbols": ["Workspace"]},
    {"slug": "sdk/model", "title": "Models", "sidebar": "Models",
     "desc": "Trained, deployable predictors.",
     "symbols": ["Model", "DiffModel"]},
    {"slug": "sdk/trainer", "title": "Trainer", "sidebar": "Trainer",
     "desc": "Configure and run general-purpose training.",
     "symbols": ["Trainer", "Metric"]},
    {"slug": "sdk/jobs", "title": "Jobs", "sidebar": "Jobs",
     "desc": "Async job handles returned by long-running SDK calls.",
     "symbols": ["Job", "TrainingJob", "JobResult", "JobFailed", "JobNotReady"]},
    {"slug": "sdk/tasks", "title": "Tasks", "sidebar": "Tasks",
     "desc": "Supervised-learning task configs (the target and its columns).",
     "symbols": ["TaskKind", "Regression", "Binclass", "Multiclass", "Forecasting",
                 "SequenceRegression", "SequenceBinclass", "SequenceMulticlass"]},
    {"slug": "sdk/hpo", "title": "Hyperparameter search", "sidebar": "HPO search",
     "desc": "Define the model-family search space and the optimizer.",
     "symbols": ["HPOSpace", "ModelParamSpace", "Float", "Int", "Categorical",
                 "Mixture", "Optimizer"]},
]


# --- reStructuredText -> Markdown/MDX -----------------------------------------

def _short(target: str) -> str:
    return target.lstrip("~").split(".")[-1].rstrip("()")


def _dedent(block: str) -> str:
    lines = block.split("\n")
    indents = [len(l) - len(l.lstrip()) for l in lines if l.strip()]
    n = min(indents) if indents else 0
    return "\n".join(l[n:] if len(l) >= n else l for l in lines).strip("\n")


def convert_rst(text: str) -> str:
    if not text:
        return ""
    text = re.sub(
        r"\.\. code-block:: *(\w+)\n\n((?:[ \t]+.*\n?)+)",
        lambda m: f"```{m.group(1)}\n{_dedent(m.group(2))}\n```\n",
        text,
    )
    text = re.sub(r":\w+:`([^`]+)`", lambda m: f"`{_short(m.group(1))}`", text)
    text = re.sub(r"(?<!`)``(?!`)", "`", text)   # rST ``x`` -> `x`, keep ``` fences
    return text.strip()


def clean_type(s) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    if len(s) >= 2 and s[0] in "\"'" and s[-1] == s[0]:  # strip forward-ref quotes
        s = s[1:-1]
    s = re.sub(r":\w+:`([^`]+)`", lambda m: _short(m.group(1)), s)
    s = s.replace("builtins.", "")
    return s.replace("``", "").replace("`", "").strip()


def esc_attr(s: str) -> str:
    return s.replace('"', "'")


# --- griffe helpers ------------------------------------------------------------

def resolve(obj):
    if getattr(obj, "is_alias", False):
        try:
            return obj.final_target
        except Exception:
            return None
    return obj


def sections(obj):
    doc = getattr(obj, "docstring", None)
    return doc.parsed if doc else []


def skind(sec) -> str:
    k = getattr(sec, "kind", "")
    return getattr(k, "value", str(k))


def get_default(p):
    for attr in ("default", "value"):
        v = getattr(p, attr, None)
        if v is not None:
            return str(v)
    return None


def is_property(obj) -> bool:
    return getattr(obj, "is_attribute", False) and "property" in (getattr(obj, "labels", set()) or set())


def docstring_text(obj) -> str:
    doc = getattr(obj, "docstring", None)
    return convert_rst(doc.value) if doc else ""


def has_doc(obj) -> bool:
    """Docstring-gating: a member is documented only if it has a docstring."""
    d = getattr(obj, "docstring", None)
    return bool(d and (d.value or "").strip())


# --- rendering -----------------------------------------------------------------

def render_signature(func, name=None) -> str:
    bits, star_done = [], False
    for p in getattr(func, "parameters", []) or []:
        kname = getattr(getattr(p, "kind", None), "name", "")
        if p.name in ("self", "cls"):
            continue
        if kname == "keyword_only" and not star_done:
            bits.append("*")
            star_done = True
        if kname == "var_positional":
            star_done = True
            bits.append("*" + p.name)
            continue
        if kname == "var_keyword":
            bits.append("**" + p.name)
            continue
        piece = p.name
        ann = getattr(p, "annotation", None)
        if ann is not None:
            piece += f": {clean_type(ann)}"
        d = get_default(p)
        if d is not None:
            piece += f" = {d}"
        bits.append(piece)
    ret = getattr(func, "returns", None)
    arrow = f" -> {clean_type(ret)}" if ret is not None else ""
    return f"{name or func.name}({', '.join(bits)}){arrow}"


def param_fields(obj) -> list[str]:
    sig_defaults = {p.name: get_default(p) for p in (getattr(obj, "parameters", []) or [])}
    out = []
    for sec in sections(obj):
        if skind(sec) != "parameters":
            continue
        for p in sec.value:
            raw_ann = getattr(p, "annotation", "") or ""
            typ = clean_type(raw_ann)
            desc = convert_rst(getattr(p, "description", "") or "")
            optional = "optional" in str(raw_ann).lower()
            has_default = sig_defaults.get(p.name) is not None
            flag = "" if (optional or has_default) else " required"
            typ_attr = f' type="{esc_attr(typ)}"' if typ else ""
            out.append(f'<ParamField body="{p.name}"{typ_attr}{flag}>\n  {desc}\n</ParamField>')
    return out


def response_fields(obj) -> list[str]:
    out = []
    for sec in sections(obj):
        if skind(sec) != "returns":
            continue
        for r in sec.value:
            typ = clean_type(getattr(r, "annotation", "") or "")
            desc = convert_rst(getattr(r, "description", "") or "")
            typ_attr = f' type="{esc_attr(typ)}"' if typ else ""
            out.append(f'<ResponseField name="returns"{typ_attr}>\n  {desc}\n</ResponseField>')
    return out


def render_attr(a) -> str:
    typ = clean_type(getattr(a, "annotation", "") or "")
    desc = docstring_text(a)
    typ_attr = f' type="{esc_attr(typ)}"' if typ else ""
    return f'<ParamField body="{a.name}"{typ_attr}>\n  {desc}\n</ParamField>'


def render_property(p) -> str:
    typ = clean_type(getattr(p, "annotation", "") or "")
    desc = docstring_text(p)
    typ_attr = f' type="{esc_attr(typ)}"' if typ else ""
    return f'<ResponseField name="{p.name}"{typ_attr}>\n  {desc}\n</ResponseField>'


def render_callable(obj, call_name, level="##") -> str:
    sub = "###" if level == "##" else "**"
    sub_end = "" if level == "##" else "**"
    parts = [f"{level} `{call_name}`\n", f"```python\n{render_signature(obj, obj.name)}\n```\n"]
    body = docstring_text(obj)
    if body:
        parts.append(body + "\n")
    pf = param_fields(obj)
    if pf:
        parts.append(f"{sub}Parameters{sub_end}\n\n" + "\n\n".join(pf) + "\n")
    rf = response_fields(obj)
    if rf:
        parts.append(f"{sub}Returns{sub_end}\n\n" + "\n\n".join(rf) + "\n")
    return "\n".join(parts)


def render_class(cls) -> str:
    parts = [f"## `{cls.name}`\n"]
    body = docstring_text(cls)
    if body:
        parts.append(body + "\n")
    pf = param_fields(cls)  # dataclass / NumPy-documented fields
    props, methods, attrs = [], [], []
    members = dict(cls.members or {})
    try:  # surface only *promoted* inherited members; skip framework plumbing
        for nm, m in (cls.inherited_members or {}).items():
            if nm in INHERITED_ALLOW:
                members.setdefault(nm, m)
    except Exception:
        pass
    for name, m in members.items():
        r = resolve(m)
        if r is None or name.startswith("_") or name in EXCLUDE or name in SKIP_ATTRS:
            continue
        # Docstring-gating for fields/properties (empty entries are pure noise);
        # methods are kept even when undocumented — a typed signature is useful.
        if is_property(r):
            if has_doc(r):
                props.append(r)
        elif getattr(r, "is_function", False):
            methods.append(r)
        elif getattr(r, "is_attribute", False):
            if has_doc(r):
                attrs.append(r)
    if pf:
        parts.append("### Fields\n\n" + "\n\n".join(pf) + "\n")
    elif attrs:
        parts.append("### Fields\n\n" + "\n\n".join(render_attr(a) for a in attrs) + "\n")
    if props:
        parts.append("### Properties\n\n" + "\n\n".join(render_property(p) for p in props) + "\n")
    for m in methods:
        parts.append(render_callable(m, f"{cls.name}.{m.name}()", level="###"))
    return "\n".join(parts)


def coverage_report(pkg):
    """Undocumented public methods per documented class — a punch-list for the
    SDK team, and a candidate CI gate ('fail if coverage drops')."""
    rows = []
    for page in PAGES:
        for name in page["symbols"]:
            if name in EXCLUDE:
                continue
            obj = resolve(pkg.members.get(name))
            if obj is None or not getattr(obj, "is_class", False):
                continue
            members = dict(obj.members or {})
            try:
                for nm, m in (obj.inherited_members or {}).items():
                    if nm in INHERITED_ALLOW:
                        members.setdefault(nm, m)
            except Exception:
                pass
            funcs = [(nm, resolve(m)) for nm, m in members.items()
                     if not nm.startswith("_") and nm not in EXCLUDE and nm not in SKIP_ATTRS]
            funcs = [(nm, r) for nm, r in funcs if r is not None and getattr(r, "is_function", False)]
            undoc = sorted(nm for nm, r in funcs if not has_doc(r))
            if funcs:
                rows.append((name, len(funcs), undoc))
    return rows


def render_symbol(pkg, name):
    member = pkg.members.get(name)
    if member is None:
        return None, "missing"
    obj = resolve(member)
    if obj is None:
        return None, "external"
    if getattr(obj, "is_class", False):
        return render_class(obj), "class"
    if getattr(obj, "is_function", False):
        return render_callable(obj, f"op.{name}()"), "function"
    return None, "other"


# --- main ---------------------------------------------------------------------

def main() -> int:
    try:
        extensions = griffe.load_extensions("griffe_pydantic")
        print("griffe-pydantic: enabled")
    except Exception as e:
        extensions = None
        print(f"  warn: griffe-pydantic not loaded ({e}); pydantic field docs may be empty", file=sys.stderr)

    loader = griffe.GriffeLoader(docstring_parser="numpy", extensions=extensions)
    pkg = loader.load(PKG)
    for extra in ("outerproduct_http_types", "outerproduct_dataframe"):
        try:  # so re-exported config types + inherited Workspace/Context methods resolve
            loader.load(extra)
        except Exception as e:
            print(f"  warn: could not load {extra}: {e}", file=sys.stderr)
    try:
        loader.resolve_aliases(implicit=False, external=True)
    except Exception as e:
        print(f"  warn: resolve_aliases: {e}", file=sys.stderr)

    total_leaks = 0
    for page in PAGES:
        body_parts, documented, missing = [], [], []
        for name in page["symbols"]:
            if name in EXCLUDE:
                continue
            md, kind = render_symbol(pkg, name)
            if md:
                body_parts.append(md)
                documented.append(name)
            else:
                missing.append(f"{name}({kind})")
        front = [
            "---",
            f'title: "{page["title"]}"',
            f'sidebarTitle: "{page["sidebar"]}"',
            f'description: "{esc_attr(page["desc"])}"',
            "---",
            "",
            "{/* AUTO-GENERATED by scripts/gen_sdk_mdx.py — do not edit by hand. */}",
            "",
        ]
        if page.get("intro"):
            front += [page["intro"], ""]
        content = "\n".join(front) + "\n".join(body_parts).rstrip() + "\n"
        path = page["slug"] + ".mdx"
        with open(path, "w") as fh:
            fh.write(content)
        leaks = [ln for ln in content.splitlines() if "reasoning" in ln.lower()]
        total_leaks += len(leaks)
        flag = f"  ⚠ {len(leaks)} reasoning mention(s)" if leaks else ""
        miss = f"  (skipped: {missing})" if missing else ""
        print(f"  {path:24} {len(documented):2} symbols{flag}{miss}")

    # nav fragment to paste into docs.json (SDK Reference tab)
    nav = {"tab": "SDK Reference", "pages": [p["slug"] for p in PAGES]}
    print("\nNav fragment for docs.json (SDK Reference tab):")
    print(json.dumps(nav, indent=2))
    print(f"\nTotal reasoning mentions across pages: {total_leaks} "
          f"({'clean' if total_leaks == 0 else 'scrub before publish'})")

    print("\nDocstring coverage (undocumented public methods per class):")
    rows = coverage_report(pkg)
    total_m = sum(n for _, n, _ in rows)
    total_u = sum(len(u) for _, _, u in rows)
    for name, n, undoc in rows:
        mark = "OK" if not undoc else f"{len(undoc)}/{n} undocumented"
        print(f"  {name:16} {mark}")
        if undoc:
            print(f"      -> {', '.join(undoc)}")
    pct = 100 * (total_m - total_u) / total_m if total_m else 100
    print(f"  coverage: {total_m - total_u}/{total_m} methods documented ({pct:.0f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
