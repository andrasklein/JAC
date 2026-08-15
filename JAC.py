#!/usr/bin/env python3
"""
JAC.py -- "Just Another Compiler" helper for compiling old C exploit/PoC
source code on Kali Linux.

The problem: old (often 15-20 year old) exploit/PoC C source files frequently
fail to compile with a modern gcc on the first try (stricter C standard
enforcement, missing headers/libs, missing 32-bit support). This script
automates the loop a human would otherwise do by hand: compile, read the
error, add a flag or install a package, recompile -- until it succeeds or
the attempt budget runs out.

Four phases:
  1. Environment recon   (gcc/clang version, 32-bit multilib, apt-file status)
  2. Static analysis     (#include -> likely -l flags, old-C patterns)
  3. Compile loop         (rule engine driven by stderr patterns)
  4. Report               (working command, or remaining errors + next steps)

Safety constraints:
  - The script NEVER silently modifies the source file. Source-level fixes
    are only applied after showing a concrete diff and getting an explicit
    (interactive) confirmation -- --assume-yes does not bypass this.
  - Packages are only installed after approval (or with --assume-yes),
    otherwise the install command is just printed.

Usage examples:
  ./JAC.py exploit.c
  ./JAC.py exploit.c -o expl --max-attempts 8
  ./JAC.py old_exploit.c --dry-run
  sudo ./JAC.py priv_esc.c --assume-yes

Uses only the Python standard library (subprocess, argparse, re, shutil, ...).
"""

import argparse
import difflib
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field

# ============================================================================
# RULE TABLES -- this is where to add a new pattern/fix
# ============================================================================
#
# The top of the file deliberately holds all the "knowledge": which header
# implies which linker flag, which old-C pattern can be handled with which
# -std flag, and which compile error message triggers which intervention.
# Adding a new rule: extend the relevant list/dict, no need to touch the
# code elsewhere (the engine walks the tables at runtime).

# --- 1) Header -> linker flag(s) mapping -----------------------------------
# When an #include shows up in the source, we try linking with these flags
# already on the first compile attempt, to avoid an unnecessary
# "undefined reference" round-trip.
INCLUDE_TO_LIBS = {
    "pthread.h": ["-lpthread"],
    "math.h": ["-lm"],
    "crypt.h": ["-lcrypt"],
    "dlfcn.h": ["-ldl"],
    "zlib.h": ["-lz"],
    "readline/readline.h": ["-lreadline"],
    "readline/history.h": ["-lhistory"],
    "curses.h": ["-lcurses"],
    "ncurses.h": ["-lncurses"],
    "openssl/ssl.h": ["-lssl", "-lcrypto"],
    "openssl/evp.h": ["-lcrypto"],
    "openssl/rsa.h": ["-lcrypto"],
    "resolv.h": ["-lresolv"],
    "rpc/rpc.h": ["-ltirpc"],
    "gmp.h": ["-lgmp"],
    "ldap.h": ["-lldap"],
}

# --- 2) "undefined reference to X" -> symbol prefix -> linker flag --------
# Fallback for when the #include list didn't reveal the lib (e.g. the source
# doesn't include the header and just hand-declares the function -- common
# in old exploits).
SYMBOL_TO_LIB = [
    (re.compile(r"^pthread_"), "-lpthread"),
    (re.compile(r"^(dlopen|dlsym|dlclose|dlerror|dladdr)$"), "-ldl"),
    (re.compile(r"^(crypt|crypt_r)$"), "-lcrypt"),
    (re.compile(r"^(sin|cos|tan|pow|sqrt|exp|log|log2|log10|floor|ceil|fabs|atan2?|round|hypot)$"), "-lm"),
    (re.compile(r"^(gzopen|gzread|gzwrite|gzclose|deflate|inflate|crc32)$"), "-lz"),
    (re.compile(r"^(readline|add_history)$"), "-lreadline"),
    (re.compile(r"^(SSL_|TLS_|EVP_|RSA_|BIO_|X509_)"), "-lssl -lcrypto"),
    (re.compile(r"^(res_init|res_query|dn_expand)$"), "-lresolv"),
]

# --- 3) Old-C patterns -> proactively suggested -std flag ------------------
# Best-effort, regex-based heuristics (not a real parser). The goal is to
# already pick a good -std in the static analysis phase, before the first
# attempt, so we don't burn an extra round on it.
#   name    : short identifier for the report
#   pattern : regex against the source
#   std     : suggested -std= value
#   note    : human-readable explanation
OLD_C_STD_HINTS = [
    (
        "knr_function_def",
        re.compile(
            r"(?m)^\w[\w\s\*]*\([^;{}\n]*\)\s*\n(?:\s*[A-Za-z_][\w\s\*]*;\s*\n)+\s*\{"
        ),
        "gnu89",
        "K&R-style function definition (parameter types below the header)",
    ),
    (
        "implicit_main",
        re.compile(r"(?m)^(?!.*\b(?:int|void|unsigned|static)\b)\s*main\s*\("),
        "gnu89",
        "'main' without a return type (implicit int)",
    ),
]

# --- 4) Detecting a 32-bit target from the source --------------------------
BITS32_HINT_PATTERNS = [
    re.compile(r"%e(ax|bx|cx|dx|si|di|bp|sp)\b"),   # 32-bit inline asm registers
    re.compile(r"int\s+\$0x80"),                     # classic 32-bit syscall trap
]
BITS64_MARKERS = re.compile(r"%r(ax|bx|cx|dx|si|di|bp|sp|8|9|10|11|12|13|14|15)\b")


# --- 5) Compile error -> intervention rule table ---------------------------
# This is the heart of the engine. Every rule is (regex, handler). The
# handler is invoked for every match found in the full stderr text, and can
# modify the next attempt's command line via the `Ctx` object (state, args,
# ...), or queue an install/patch suggestion. The handler returns a short,
# human-readable string (what it did), or None if there was nothing new to
# do (e.g. already tried).
#
# ADDING A NEW RULE: append a (regex, handler) pair to this list.
# Handler signature: handler(match, ctx) -> str | None


def h_missing_header(match, ctx):
    header = match.group(1)
    key = ("header", header)
    if key in ctx.state.tried_header_lookups:
        return None
    ctx.state.tried_header_lookups.add(key)

    # special case: missing 32-bit multilib header alongside -m32
    if header in ("gnu/stubs-32.h", "bits/libc-header-start.h") or (
        "-m32" in ctx.state.pre_flags and header.startswith("bits/")
    ):
        return suggest_multilib(ctx)

    packages = apt_file_search(header, ctx, path_hint="/usr/include/")
    return propose_install(
        f"Missing header: {header}", packages, ctx, manual_note=f"#include of missing header: {header}"
    )


def h_undefined_reference(match, ctx):
    symbol = match.group(1)
    key = ("symbol", symbol)
    if key in ctx.state.tried_symbol_lookups:
        return None
    ctx.state.tried_symbol_lookups.add(key)

    # gets() -- fully removed from glibc since 2.26+, no library brings it
    # back by linking: this can ONLY be fixed at the source level.
    if symbol == "gets":
        return queue_gets_patch(ctx)

    for pattern, libflag in SYMBOL_TO_LIB:
        if pattern.match(symbol):
            added = False
            for lf in libflag.split():
                if add_lib(ctx.state, lf):
                    added = True
            return f"undefined reference to `{symbol}' -> added {libflag}" if added else None

    ctx.state.unresolved.append(
        f"undefined reference to `{symbol}': no known automatic mapping, "
        f"add the right -l flag by hand, or check for a typo."
    )
    return None


def h_cannot_find_lib(match, ctx):
    libname = match.group(1)
    key = ("lib", libname)
    if key in ctx.state.tried_lib_lookups:
        return None
    ctx.state.tried_lib_lookups.add(key)

    packages = apt_file_search(f"lib{libname}.so", ctx, path_hint=".so")
    return propose_install(
        f"Missing shared library: -l{libname} (lib{libname}.so not found)",
        packages,
        ctx,
        manual_note=f"cannot find -l{libname}",
    )


def h_implicit_function_or_int(match, ctx):
    key = "implicit_decl"
    if key in ctx.state.flag_fix_applied:
        return None
    ctx.state.flag_fix_applied.add(key)
    changed = []
    if add_flag(ctx.state, "-Wno-implicit-function-declaration"):
        changed.append("-Wno-implicit-function-declaration")
    if add_flag(ctx.state, "-Wno-implicit-int"):
        changed.append("-Wno-implicit-int")
    if add_flag(ctx.state, "-fpermissive"):
        changed.append("-fpermissive")
    if ctx.state.std_flag not in ("-std=gnu89", "-std=gnu99"):
        ctx.state.std_flag = "-std=gnu89"
        changed.append("-std=gnu89")
    return "implicit function/int error -> " + ", ".join(changed) if changed else None


def h_incompatible_pointer_types(match, ctx):
    # gcc 14+ (and clang) promoted these from warnings to hard errors by
    # default. Extremely common in older exploits: a callback (clone(),
    # pthread_create(), signal(), qsort(), ...) declared with an empty/
    # old-style parameter list, e.g. 'static int child() {' instead of
    # 'static int child(void *arg) {'. Pre-gcc14 this compiled fine as a
    # warning, so demoting it back to a warning reproduces that behavior
    # without touching the source.
    key = "incompatible_pointer_types"
    if key in ctx.state.flag_fix_applied:
        return None
    ctx.state.flag_fix_applied.add(key)
    changed = []
    if add_flag(ctx.state, "-Wno-incompatible-pointer-types"):
        changed.append("-Wno-incompatible-pointer-types")
    if add_flag(ctx.state, "-Wno-int-conversion"):
        changed.append("-Wno-int-conversion")
    return "incompatible pointer type error -> " + ", ".join(changed) if changed else None


def h_c99_for_loop(match, ctx):
    key = "c99_for_loop"
    if key in ctx.state.flag_fix_applied:
        return None
    ctx.state.flag_fix_applied.add(key)
    ctx.state.std_flag = "-std=gnu99"
    return "'for' loop declaration requires C99 -> -std=gnu99"


def h_knr_old_style(match, ctx):
    key = "knr_old_style"
    if key in ctx.state.flag_fix_applied:
        return None
    ctx.state.flag_fix_applied.add(key)
    # this can't be fixed with a flag: needs a source-level rewrite
    return queue_knr_patch(ctx)


def h_m32_multilib(match, ctx):
    return suggest_multilib(ctx)


# The rule table itself -- ORDER: the engine tries all matches top to
# bottom, and every rule runs over the full stderr text (finditer), so
# several different errors can be fixed in a single round.
ERROR_RULES = [
    (
        re.compile(r"fatal error:\s*([\w./+-]+):\s*No such file or directory"),
        h_missing_header,
    ),
    (
        re.compile(r"undefined reference to [`'‘]([\w:.]+)['’]"),
        h_undefined_reference,
    ),
    (
        re.compile(r"cannot find -l([\w.+-]+)"),
        h_cannot_find_lib,
    ),
    (
        re.compile(r"implicit declaration of function|type defaults to .int.|return type defaults to .int.|-Wimplicit-int"),
        h_implicit_function_or_int,
    ),
    (
        re.compile(r"\[-Wincompatible-pointer-types\]|\[-Wincompatible-function-pointer-types\]|\[-Wint-conversion\]"),
        h_incompatible_pointer_types,
    ),
    (
        re.compile(r".for.\s*loop initial declarations are only allowed in C99"),
        h_c99_for_loop,
    ),
    (
        re.compile(r"old-style parameter declarations in prototyped function definition|parameter names \(without types\) in function declaration"),
        h_knr_old_style,
    ),
    (
        re.compile(r"unrecognized (?:command.line|command-line) option .-m32.|as: unrecognized option '--32'"),
        h_m32_multilib,
    ),
]

# Exploit-typical flags: included in the first attempt by default, since
# they're almost always needed for old PoCs and rarely break an otherwise
# correct compile on their own.
EXPLOIT_TYPICAL_PRE_FLAGS = ["-fno-stack-protector", "-z", "execstack", "-static"]


# ============================================================================
# STATE / CONTEXT
# ============================================================================


@dataclass
class CompileState:
    pre_flags: list = field(default_factory=list)   # -m32, -fno-stack-protector, -Wno-..., -static, ...
    libs: list = field(default_factory=list)         # -lpthread, -lm, ... (must come AFTER the source)
    std_flag: str = None                              # only one can be active at a time

    tried_header_lookups: set = field(default_factory=set)
    tried_symbol_lookups: set = field(default_factory=set)
    tried_lib_lookups: set = field(default_factory=set)
    tried_installs: set = field(default_factory=set)
    flag_fix_applied: set = field(default_factory=set)
    declined_patches: set = field(default_factory=set)
    applied_patches: list = field(default_factory=list)

    unresolved: list = field(default_factory=list)   # manual to-dos that go into the final report


@dataclass
class Ctx:
    state: CompileState
    args: argparse.Namespace
    source_path: str
    apt_file_ok: bool


def add_flag(state, flag):
    if flag not in state.pre_flags:
        state.pre_flags.append(flag)
        return True
    return False


def add_lib(state, libflag):
    if libflag not in state.libs:
        state.libs.append(libflag)
        return True
    return False


def suggest_multilib(ctx):
    key = "gcc-multilib"
    if key in ctx.state.tried_installs:
        return None
    packages = ["gcc-multilib", "libc6-dev-i386"]
    return propose_install(
        "Headers/libs needed for a 32-bit target are missing (-m32)",
        packages,
        ctx,
        manual_note="multilib support is missing for -m32 compilation",
        install_key=key,
    )


# ============================================================================
# APT-FILE / PACKAGE INSTALLATION
# ============================================================================


def run(cmd, timeout=30, capture=True, text=True):
    try:
        return subprocess.run(cmd, capture_output=capture, text=text, timeout=timeout)
    except FileNotFoundError:
        return None
    except subprocess.TimeoutExpired:
        return None


def apt_file_status():
    """(available: bool, ready: bool, message: str)"""
    if not shutil.which("apt-file"):
        return False, False, "apt-file is not installed (sudo apt-get install apt-file && sudo apt-file update)"
    proc = run(["apt-file", "search", "stdio.h"], timeout=15)
    if proc is None:
        return True, False, "calling apt-file failed / timed out"
    if proc.returncode != 0 or "You need to give the update command first" in (proc.stderr or ""):
        return True, False, "the apt-file cache is missing/stale (run: sudo apt-file update)"
    return True, True, "apt-file is available and up to date"


def apt_file_search(term, ctx, path_hint=None, limit=5):
    if not ctx.apt_file_ok:
        return []
    proc = run(["apt-file", "search", term], timeout=20)
    if proc is None or proc.returncode != 0:
        return []
    packages = []
    hinted, plain = [], []
    for line in proc.stdout.splitlines():
        if ":" not in line:
            continue
        pkg, _, path = line.partition(":")
        pkg = pkg.strip()
        if not pkg:
            continue
        target = hinted if (path_hint and path_hint in path) else plain
        if pkg not in target:
            target.append(pkg)
    packages = hinted + [p for p in plain if p not in hinted]
    return packages[:limit]


def confirm(prompt, assume_yes):
    if assume_yes:
        print(f"[assume-yes] {prompt} -> yes")
        return True
    try:
        resp = input(f"{prompt} [y/N]: ").strip().lower()
    except EOFError:
        resp = ""
    return resp in ("y", "yes")


def propose_install(reason, packages, ctx, manual_note=None, install_key=None):
    print(f"  [!] {reason}")
    if not packages:
        msg = manual_note or reason
        if not ctx.apt_file_ok:
            ctx.state.unresolved.append(
                f"{msg} -- apt-file is not available/not up to date, could not look up a package name. "
                f"Look it up by hand: sudo apt-file update && apt-file search <filename>"
            )
        else:
            ctx.state.unresolved.append(f"{msg} -- apt-file found no matching package, look it up by hand.")
        return None

    key = install_key or tuple(sorted(packages))
    if key in ctx.state.tried_installs:
        return None
    ctx.state.tried_installs.add(key)

    prefix = [] if os.geteuid() == 0 else ["sudo"]
    install_cmd = prefix + ["apt-get", "install", "-y"] + packages
    print(f"      suggested package(s): {' '.join(packages)}")
    print(f"      install command:      {' '.join(install_cmd)}")

    if confirm("      Install now?", ctx.args.assume_yes):
        proc = subprocess.run(install_cmd)
        if proc.returncode == 0:
            print("      -> installed.")
            return f"installed: {' '.join(packages)}"
        ctx.state.unresolved.append(f"{reason} -- install failed ({' '.join(install_cmd)})")
        return None
    else:
        ctx.state.unresolved.append(
            f"{manual_note or reason} -- not installed, run by hand: {' '.join(install_cmd)}"
        )
        return None


# ============================================================================
# SOURCE-LEVEL PATCH SUGGESTIONS (never automatic!)
# ============================================================================


def confirm_patch(description):
    """Approval for a source modification -- --assume-yes NEVER bypasses
    this, because silently modifying the source of an exploit can break it
    in ways that go unnoticed."""
    print("  [PATCH SUGGESTION]")
    print(description)
    try:
        resp = input("      Apply this source modification? [y/N]: ").strip().lower()
    except EOFError:
        resp = ""
    return resp in ("y", "yes")


def _read_source(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _write_source_with_backup(path, new_text):
    backup = path + ".bak"
    if not os.path.exists(backup):
        shutil.copy2(path, backup)
    with open(path, "w", encoding="utf-8") as f:
        f.write(new_text)
    print(f"      original saved to: {backup}")


def queue_gets_patch(ctx):
    key = "gets_to_fgets"
    if key in ctx.state.declined_patches or key in ctx.state.applied_patches:
        return None
    text = _read_source(ctx.source_path)
    pattern = re.compile(r"\bgets\s*\(\s*([A-Za-z_]\w*)\s*\)")
    m = pattern.search(text)
    if not m:
        ctx.state.unresolved.append(
            "undefined reference to `gets': gets() has been fully removed from glibc, "
            "replace it by hand with fgets(buf, sizeof(buf), stdin)."
        )
        return None
    old_line = m.group(0)
    varname = m.group(1)
    new_line = f"fgets({varname}, sizeof({varname}), stdin)"
    new_text = text[: m.start()] + new_line + text[m.end():]
    diff = "\n".join(
        difflib.unified_diff(
            [old_line + "\n"], [new_line + "\n"],
            fromfile=ctx.source_path, tofile=ctx.source_path, lineterm="",
        )
    )
    description = (
        f"      reason: gets() does not exist in modern glibc, no -l flag brings it back.\n"
        f"{diff}"
    )
    if confirm_patch(description):
        _write_source_with_backup(ctx.source_path, new_text)
        ctx.state.applied_patches.append(key)
        return "gets() -> fgets() source patch applied"
    else:
        ctx.state.declined_patches.add(key)
        ctx.state.unresolved.append(
            f"the gets() call needs to be replaced (declined), by hand:\n{diff}"
        )
        return None


_KNR_PATTERN = re.compile(
    r"(?P<name>[A-Za-z_]\w*)\s*\((?P<params>[^)]*)\)\s*\n"
    r"(?P<decls>(?:\s*[A-Za-z_][\w\s\*]*;\s*\n)+)"
    r"\s*\{"
)


def _try_build_knr_patch(text):
    for m in _KNR_PATTERN.finditer(text):
        params = [p.strip() for p in m.group("params").split(",") if p.strip()]
        if not params:
            continue
        decl_lines = [d.strip() for d in m.group("decls").strip().splitlines()]
        type_by_name = {}
        ok = True
        for decl in decl_lines:
            decl = decl.rstrip(";").strip()
            mm = re.match(r"^(?P<type>[A-Za-z_][\w\s]*?)\s*(?P<stars>\**)\s*(?P<name>[A-Za-z_]\w*)$", decl)
            if not mm:
                ok = False
                break
            type_by_name[mm.group("name")] = f"{mm.group('type').strip()} {mm.group('stars')}".strip()
        if not ok or not all(p in type_by_name for p in params):
            continue
        new_params = ", ".join(f"{type_by_name[p]} {p}" for p in params)
        old_snippet = m.group(0)
        new_snippet = f"{m.group('name')}({new_params})\n{{"
        return old_snippet, new_snippet
    return None


def queue_knr_patch(ctx):
    key = "knr_signature"
    if key in ctx.state.declined_patches or key in ctx.state.applied_patches:
        return None
    text = _read_source(ctx.source_path)
    result = _try_build_knr_patch(text)
    if result is None:
        ctx.state.unresolved.append(
            "K&R-style (old-style) function parameter declaration -- could not automatically "
            "generate a safe diff, rewrite the function header as an ANSI prototype by hand, "
            "e.g.: 'foo(a, b)\\n  int a;\\n  char *b;\\n{' -> 'foo(int a, char *b)\\n{'"
        )
        return None
    old_snippet, new_snippet = result
    new_text = text.replace(old_snippet, new_snippet, 1)
    diff = "\n".join(
        difflib.unified_diff(
            old_snippet.splitlines(keepends=True),
            new_snippet.splitlines(keepends=True),
            fromfile=ctx.source_path, tofile=ctx.source_path, lineterm="",
        )
    )
    description = (
        "      reason: gcc 14+ treats mixed prototype/K&R parameter declarations as an error, "
        "this can only be fixed by rewriting the source, not with a flag.\n"
        f"{diff}"
    )
    if confirm_patch(description):
        _write_source_with_backup(ctx.source_path, new_text)
        ctx.state.applied_patches.append(key)
        return "K&R function header -> ANSI prototype patch applied"
    else:
        ctx.state.declined_patches.add(key)
        ctx.state.unresolved.append(f"K&R function header rewrite needed (declined), by hand:\n{diff}")
        return None


# ============================================================================
# PHASE 1: ENVIRONMENT RECON
# ============================================================================


def phase_env_recon():
    print("== Phase 1: environment recon ==")
    compiler = None
    for candidate in ("gcc", "clang"):
        if shutil.which(candidate):
            proc = run([candidate, "--version"], timeout=10)
            version_line = proc.stdout.splitlines()[0] if proc and proc.stdout else "(unknown version)"
            print(f"  {candidate}: {version_line}")
            if compiler is None:
                compiler = candidate
        else:
            print(f"  {candidate}: not installed")

    if compiler is None:
        print("  [ERROR] neither gcc nor clang was found -- install one: sudo apt-get install gcc")
        return None, False

    multilib_ok = pkg_installed("gcc-multilib") and pkg_installed("libc6-dev-i386")
    print(f"  32-bit (gcc-multilib/libc6-dev-i386) support: {'yes' if multilib_ok else 'no (or could not be checked)'}")

    apt_file_available, apt_file_ready, apt_file_msg = apt_file_status()
    if not apt_file_ready:
        print(f"  [WARNING] {apt_file_msg}")
        print("            without it, missing header/lib -> package name lookup won't work automatically.")
    else:
        print(f"  apt-file: {apt_file_msg}")

    return compiler, apt_file_ready


def pkg_installed(name):
    proc = run(["dpkg", "-s", name], timeout=10)
    return proc is not None and proc.returncode == 0


# ============================================================================
# PHASE 2: STATIC ANALYSIS
# ============================================================================


INCLUDE_RE = re.compile(r'^\s*#\s*include\s*[<"]([^">]+)[>"]', re.MULTILINE)


def phase_static_analysis(source_text):
    print("== Phase 2: static analysis ==")

    includes = INCLUDE_RE.findall(source_text)
    print(f"  #include found: {', '.join(includes) if includes else '(none)'}")

    libs = []
    for inc in includes:
        for lf in INCLUDE_TO_LIBS.get(inc, []):
            if lf not in libs:
                libs.append(lf)
    if libs:
        print(f"  derived linker flags: {' '.join(libs)}")

    std_hint = None
    for name, pattern, std, note in OLD_C_STD_HINTS:
        if pattern.search(source_text):
            print(f"  old-C pattern ({name}): {note} -> suggested -std={std}")
            std_hint = std
            break  # the first match is enough to pick the starting -std

    bits32 = any(p.search(source_text) for p in BITS32_HINT_PATTERNS) and not BITS64_MARKERS.search(source_text)
    if bits32:
        print("  suspected 32-bit target (inline asm 32-bit registers / int $0x80) -> -m32 added to the attempt")

    return libs, std_hint, bits32


# ============================================================================
# PHASE 3: COMPILE LOOP
# ============================================================================


def build_command(compiler, source_path, output_path, state):
    cmd = [compiler] + state.pre_flags
    if state.std_flag:
        cmd.append(state.std_flag)
    cmd += ["-o", output_path, source_path]
    cmd += state.libs  # -l flags must come AFTER the source (linker order)
    return cmd


def try_compile(cmd, timeout=60):
    proc = run(cmd, timeout=timeout)
    if proc is None:
        return False, "(calling the compiler failed or timed out)"
    return proc.returncode == 0, proc.stderr or ""


def apply_error_rules(stderr_text, ctx):
    actions = []
    for pattern, handler in ERROR_RULES:
        for m in pattern.finditer(stderr_text):
            result = handler(m, ctx)
            if result:
                actions.append(result)
    return actions


def phase_compile_loop(compiler, source_path, output_path, state, args, apt_file_ok):
    print("== Phase 3: compile loop ==")
    ctx = Ctx(state=state, args=args, source_path=source_path, apt_file_ok=apt_file_ok)

    last_stderr = ""
    last_flags_snapshot = None

    for attempt in range(1, args.max_attempts + 1):
        cmd = build_command(compiler, source_path, output_path, state)
        print(f"\n  [{attempt}/{args.max_attempts}] {' '.join(cmd)}")

        success, stderr_text = try_compile(cmd)
        last_stderr = stderr_text

        if success:
            print("  -> compiled successfully.")
            return True, cmd, ""

        print("  -> failed, analyzing error patterns...")
        actions = apply_error_rules(stderr_text, ctx)
        for a in actions:
            print(f"     * {a}")

        snapshot = (tuple(state.pre_flags), tuple(state.libs), state.std_flag)
        if not actions and snapshot == last_flags_snapshot:
            print("  -> no more automatic fixes available, stopping the loop.")
            break
        last_flags_snapshot = snapshot

    return False, build_command(compiler, source_path, output_path, state), last_stderr


# ============================================================================
# PHASE 4: REPORT
# ============================================================================


def phase_report(success, cmd, stderr_text, state):
    print("\n== Phase 4: report ==")
    if success:
        print("  Compiled successfully. Full working command (usable by hand next time too):\n")
        print(f"    {' '.join(cmd)}\n")
        if state.applied_patches:
            print(f"  Applied source patch(es): {', '.join(state.applied_patches)} (original saved in a .bak file)")
        return 0

    print("  Failed to compile within the given attempt budget.\n")
    print("  Last command:")
    print(f"    {' '.join(cmd)}\n")
    if stderr_text:
        print("  Last error message (excerpt):")
        lines = stderr_text.strip().splitlines()
        for line in lines[-40:]:
            print(f"    {line}")
        print()

    if state.unresolved:
        print("  Manual to-dos:")
        for item in state.unresolved:
            print(f"    - {item}")
    else:
        print("  No concrete automatic suggestion for the remaining error, review the message above by hand.")
    return 1


# ============================================================================
# MAIN
# ============================================================================


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="JAC.py",
        description="Helper tool that automates compiling old C exploit/PoC source code on Kali Linux.",
        epilog=(
            "Examples:\n"
            "  %(prog)s exploit.c\n"
            "  %(prog)s exploit.c -o expl --max-attempts 8\n"
            "  %(prog)s old_exploit.c --dry-run\n"
            "  sudo %(prog)s priv_esc.c --assume-yes\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("source", help="path to the .c file to compile")
    parser.add_argument("-o", "--output", help="output binary name (defaults to the source name without extension)")
    parser.add_argument("--max-attempts", type=int, default=5, help="max. number of compile attempts (default 5)")
    parser.add_argument("--dry-run", action="store_true", help="only print the first suggested command, don't run it")
    parser.add_argument("--assume-yes", action="store_true", help="automatically accept package install suggestions (does NOT apply to source patches)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if not os.path.isfile(args.source):
        print(f"[ERROR] source file not found: {args.source}", file=sys.stderr)
        return 2

    output_path = args.output or os.path.splitext(args.source)[0]

    compiler, apt_file_ok = phase_env_recon()
    if compiler is None:
        return 2

    source_text = _read_source(args.source)
    libs, std_hint, bits32 = phase_static_analysis(source_text)

    state = CompileState()
    state.libs = list(libs)
    state.std_flag = f"-std={std_hint}" if std_hint else None
    state.pre_flags = list(EXPLOIT_TYPICAL_PRE_FLAGS)
    if bits32:
        state.pre_flags.insert(0, "-m32")

    if args.dry_run:
        cmd = build_command(compiler, args.source, output_path, state)
        print("\n== dry-run: suggested first command (not executed) ==")
        print(f"  {' '.join(cmd)}")
        return 0

    try:
        success, cmd, stderr_text = phase_compile_loop(
            compiler, args.source, output_path, state, args, apt_file_ok
        )
    except KeyboardInterrupt:
        print("\n[Interrupted.]")
        return 130

    return phase_report(success, cmd, stderr_text, state)


if __name__ == "__main__":
    sys.exit(main())
