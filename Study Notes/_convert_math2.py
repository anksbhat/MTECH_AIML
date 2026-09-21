import re
import sys

SUP_DIGITS = {'0':'⁰','1':'¹','2':'²','3':'³','4':'⁴','5':'⁵','6':'⁶','7':'⁷','8':'⁸','9':'⁹','+':'⁺','-':'⁻'}
SUB_DIGITS = {'0':'₀','1':'₁','2':'₂','3':'₃','4':'₄','5':'₅','6':'₆','7':'₇','8':'₈','9':'₉','+':'₊','-':'₋'}

GREEK = {
    'alpha':'α','beta':'β','gamma':'γ','delta':'δ','epsilon':'ε','varepsilon':'ε',
    'zeta':'ζ','eta':'η','theta':'θ','vartheta':'θ','iota':'ι','kappa':'κ','lambda':'λ',
    'mu':'μ','nu':'ν','xi':'ξ','pi':'π','rho':'ρ','sigma':'σ','tau':'τ','upsilon':'υ',
    'phi':'φ','varphi':'φ','chi':'χ','psi':'ψ','omega':'ω',
    'Gamma':'Γ','Delta':'Δ','Theta':'Θ','Lambda':'Λ','Xi':'Ξ','Pi':'Π','Sigma':'Σ',
    'Upsilon':'Υ','Phi':'Φ','Psi':'Ψ','Omega':'Ω',
}

def _extract_balanced(s, start):
    """s[start] must be '{'. Returns (content, index_after_closing_brace)."""
    depth = 0
    i = start
    while i < len(s):
        if s[i] == '{':
            depth += 1
        elif s[i] == '}':
            depth -= 1
            if depth == 0:
                return s[start+1:i], i + 1
        i += 1
    return s[start+1:], len(s)  # unterminated fallback

def repl_frac_sqrt(text):
    """Resolve \\frac/\\dfrac/\\tfrac and \\sqrt using a brace-depth-aware
    scanner (handles nesting like \\frac{\\eta}{\\sqrt{s_t}+\\epsilon} or
    \\frac{\\partial L}{\\partial z^{[l]}}, which plain non-nested regexes miss)."""
    def one_pass(s):
        out = []
        i = 0
        changed = False
        while i < len(s):
            m = re.match(r'\\(?:d|t)?frac', s[i:])
            if m and s[i+m.end():i+m.end()+1] == '{':
                j = i + m.end()
                num, j2 = _extract_balanced(s, j)
                if s[j2:j2+1] == '{':
                    den, j3 = _extract_balanced(s, j2)
                    out.append(f'({num})/({den})')
                    i = j3
                    changed = True
                    continue
            m2 = re.match(r'\\sqrt', s[i:])
            if m2 and s[i+m2.end():i+m2.end()+1] == '{':
                j = i + m2.end()
                arg, j2 = _extract_balanced(s, j)
                out.append(f'\u221a({arg})')
                i = j2
                changed = True
                continue
            out.append(s[i])
            i += 1
        return ''.join(out), changed

    changed = True
    while changed:
        text, changed = one_pass(text)
    # shorthand \frac1N , \tfrac12 (single-token numerator+denominator, no braces)
    text = re.sub(r'\\[dt]?frac([0-9a-zA-Z])([0-9a-zA-Z])', r'(\1)/(\2)', text)
    text = re.sub(r'\\sqrt\b', '\u221a', text)
    return text

def superscriptify(m):
    body = m.group(1)
    if all(c in SUP_DIGITS for c in body):
        return ''.join(SUP_DIGITS[c] for c in body)
    if body.startswith('[') and body.endswith(']'):
        return '^' + body
    return '^(' + body + ')' if len(body) > 1 else '^' + body

def subscriptify(m):
    body = m.group(1)
    if all(c in SUB_DIGITS for c in body):
        return ''.join(SUB_DIGITS[c] for c in body)
    if body.startswith('[') and body.endswith(']'):
        return '_' + body
    return '_(' + body + ')' if len(body) > 1 else '_' + body

def convert_cases(text):
    """Convert \\begin{cases} val & cond \\\\ ... \\end{cases} into a readable
    inline piecewise list: "{ val if cond ; val2 if cond2 }"."""
    pattern = re.compile(r'\\begin\{cases\}(.*?)\\end\{cases\}', re.S)
    def repl(m):
        body = m.group(1)
        rows = re.split(r'\\\\', body)
        parts = []
        for row in rows:
            row = row.strip()
            if not row:
                continue
            if '&' in row:
                val, cond = row.split('&', 1)
                parts.append(f'{val.strip()} if {cond.strip()}')
            else:
                parts.append(row)
        return '{ ' + ' ;  '.join(parts) + ' }'
    return pattern.sub(repl, text)

def convert_matrix(text):
    """Convert \\begin{bmatrix}...\\end{bmatrix} (and pmatrix/matrix/vmatrix)
    into compact bracket notation: rows separated by ';', columns by ','."""
    pattern = re.compile(r'\\begin\{(bmatrix|pmatrix|matrix|vmatrix)\}(.*?)\\end\{\1\}', re.S)
    def repl(m):
        body = m.group(2)
        rows = re.split(r'\\\\', body)
        row_strs = []
        for row in rows:
            row = row.strip()
            if row == '':
                continue
            cols = [c.strip() for c in row.split('&')]
            row_strs.append(', '.join(cols))
        return '[' + ';  '.join(row_strs) + ']'
    return pattern.sub(repl, text)

def convert_math(text):
    t = text

    t = convert_cases(t)
    t = convert_matrix(t)
    t = repl_frac_sqrt(t)

    # delimiters
    t = re.sub(r'\\[bB]igg?[lr]?\(', '(', t)
    t = re.sub(r'\\[bB]igg?[lr]?\)', ')', t)
    t = re.sub(r'\\left\\lfloor', '\u230a', t)
    t = re.sub(r'\\right\\rfloor', '\u230b', t)
    t = re.sub(r'\\lfloor', '\u230a', t)
    t = re.sub(r'\\rfloor', '\u230b', t)
    t = re.sub(r'\\left\\lceil', '\u2308', t)
    t = re.sub(r'\\right\\rceil', '\u2309', t)
    t = re.sub(r'\\left\\\|', '\u2016', t)
    t = re.sub(r'\\right\\\|', '\u2016', t)
    t = re.sub(r'\\left\\\{', '{', t)
    t = re.sub(r'\\right\\\}', '}', t)
    t = re.sub(r'\\left\(', '(', t)
    t = re.sub(r'\\right\)', ')', t)
    t = re.sub(r'\\left\[', '[', t)
    t = re.sub(r'\\right\]', ']', t)
    t = re.sub(r'\\left\|', '|', t)
    t = re.sub(r'\\right\|', '|', t)
    t = re.sub(r'\\left\.', '', t)
    t = re.sub(r'\\right\.', '', t)
    t = re.sub(r'\\left(?!arrow)', '', t)
    t = re.sub(r'\\right(?!arrow)', '', t)
    # standalone literal-brace set notation \{ ... \} -- protect with placeholders
    # so the later "collapse redundant braces" pass doesn't eat them.
    t = t.replace('\\{', '\u0001').replace('\\}', '\u0002')

    t = re.sub(r'\\boxed\{([^{}]*)\}', r'[\1]', t)
    t = re.sub(r'\\text\{([^{}]*)\}', r'\1', t)
    t = re.sub(r'\\mathrm\{([^{}]*)\}', r'\1', t)
    t = re.sub(r'\\mathbf\{([^{}]*)\}', r'\1', t)
    t = re.sub(r'\\mathbb\{R\}', '\u211d', t)
    t = re.sub(r'\\mathcal\{N\}', '\U0001D4A9', t)
    t = re.sub(r'\\mathcal\{B\}', '\U0001D4B7', t)
    t = re.sub(r'\\mathcal\{([^{}]*)\}', r'\1', t)

    t = re.sub(r"\\hat\{([a-zA-Z])\}", lambda m: m.group(1) + '\u0302', t)
    t = re.sub(r"\\bar\{([a-zA-Z])\}", lambda m: m.group(1) + '\u0304', t)
    t = re.sub(r"\\tilde\{([a-zA-Z])\}", lambda m: m.group(1) + '\u0303', t)
    t = re.sub(r"\\hat ([a-zA-Z])", lambda m: m.group(1) + '\u0302', t)

    # Greek letters early (so subscripts like _\theta convert cleanly downstream)
    # NOTE: use negative lookahead for a letter (not \b) because '_' counts as a
    # word character in regex, so \btheta\b would fail to match "\theta_t".
    for name, sym in sorted(GREEK.items(), key=lambda kv: -len(kv[0])):
        t = re.sub(r'\\' + name + r'(?![a-zA-Z])', sym, t)

    # arg min / arg max  (\arg\min_X or \arg\min_{X})
    t = re.sub(r'\\arg\\?min_\{([^{}]*)\}', r'argmin(\1)', t)
    t = re.sub(r'\\arg\\?min_([^\s\\{}()]+)', r'argmin(\1)', t)
    t = re.sub(r'\\arg\\?min\b', 'argmin', t)
    t = re.sub(r'\\arg\\?max_\{([^{}]*)\}', r'argmax(\1)', t)
    t = re.sub(r'\\arg\\?max_([^\s\\{}()]+)', r'argmax(\1)', t)
    t = re.sub(r'\\arg\\?max\b', 'argmax', t)

    # sum / prod / int with sub+sup limits, braced or bare token
    def limits(cmd, symbol, t):
        t = re.sub(r'\\'+cmd+r'_\{([^{}]*)\}\^\{([^{}]*)\}', lambda m: f'{symbol}({m.group(1)}..{m.group(2)})', t)
        t = re.sub(r'\\'+cmd+r'_\{([^{}]*)\}\^([^\s\\{}()]+)', lambda m: f'{symbol}({m.group(1)}..{m.group(2)})', t)
        t = re.sub(r'\\'+cmd+r'_([^\s\\{}()]+)\^\{([^{}]*)\}', lambda m: f'{symbol}({m.group(1)}..{m.group(2)})', t)
        t = re.sub(r'\\'+cmd+r'_([^\s\\{}()]+)\^([^\s\\{}()]+)', lambda m: f'{symbol}({m.group(1)}..{m.group(2)})', t)
        t = re.sub(r'\\'+cmd+r'_\{([^{}]*)\}', lambda m: f'{symbol}({m.group(1)})', t)
        t = re.sub(r'\\'+cmd+r'_([^\s\\{}()]+)', lambda m: f'{symbol}({m.group(1)})', t)
        t = re.sub(r'\\'+cmd+r'\b', symbol, t)
        return t
    t = limits('sum', 'Σ', t)
    t = limits('prod', 'Π', t)
    t = limits('int', '∫', t)
    t = re.sub(r'\\lim_\{([^{}]*)\}', r'lim(\1)', t)
    t = re.sub(r'\\lim_([^\s\\{}()]+)', r'lim(\1)', t)

    t = t.replace('\\partial', '\u2202')
    t = t.replace('\\nabla', '\u2207')
    t = t.replace('\\infty', '\u221e')
    t = t.replace('\\top', '\u1d40')
    t = t.replace('\\prime', '\u2032')
    t = t.replace('\\cdot', '\u00b7')
    t = t.replace('\\times', '\u00d7')
    t = t.replace('\\div', '\u00f7')
    t = t.replace('\\pm', '\u00b1')
    t = t.replace('\\approx', '\u2248')
    t = t.replace('\\neq', '\u2260')
    # NOTE: arrow commands (rightarrow/leftarrow/Rightarrow/...) MUST be
    # replaced before the short \le / \ge, otherwise e.g. "\leftarrow" gets
    # mangled into "\u2264ftarrow" because \le is a literal prefix of it.
    t = t.replace('\\Rightarrow', '\u21d2')
    t = t.replace('\\Leftrightarrow', '\u21d4')
    t = t.replace('\\rightarrow', '\u2192')
    t = t.replace('\\to', '\u2192')
    t = t.replace('\\leftarrow', '\u2190')
    t = t.replace('\\Leftarrow', '\u21d0')
    t = t.replace('\\leq', '\u2264')
    t = t.replace('\\geq', '\u2265')
    t = t.replace('\\le', '\u2264')
    t = t.replace('\\ge', '\u2265')
    t = t.replace('\\ll', '\u226a')
    t = t.replace('\\gg', '\u226b')
    t = t.replace('\\notin', '\u2209')
    t = t.replace('\\in', '\u2208')
    t = t.replace('\\subset', '\u2282')
    t = t.replace('\\forall', '\u2200')
    t = t.replace('\\exists', '\u2203')
    t = t.replace('\\sim', '~')
    t = t.replace('\\propto', '\u221d')
    t = t.replace('\\ldots', '\u2026')
    t = t.replace('\\dots', '\u2026')
    t = t.replace('\\cdots', '\u22ef')
    t = t.replace('\\vdots', '\u22ee')
    t = t.replace('\\odot', '\u2299')
    t = t.replace('\\oplus', '\u2295')
    t = t.replace('\\otimes', '\u2297')
    t = t.replace('\\|', '\u2016')
    t = t.replace('\\%', '%')
    t = t.replace('\\;', ' ')
    t = t.replace('\\,', ' ')
    t = t.replace('\\!', '')
    t = t.replace('\\quad', '  ')
    t = t.replace('\\qquad', '    ')
    t = t.replace('\\big', '')
    t = t.replace('\\Big', '')
    t = t.replace('\\min', ' min')
    t = t.replace('\\max', ' max')
    t = t.replace('\\log', ' log')
    t = t.replace('\\exp', ' exp')
    t = t.replace('\\sign', ' sign')

    # superscripts / subscripts: braced first, then bare single token
    t = re.sub(r'\^\{([^{}]*)\}', superscriptify, t)
    t = re.sub(r'_\{([^{}]*)\}', subscriptify, t)
    t = re.sub(r'\^([0-9+\-])(?![a-zA-Z0-9])', superscriptify, t)
    t = re.sub(r'_([0-9+\-])(?![a-zA-Z0-9])', subscriptify, t)
    # remaining bare ^token / _token (letters, e.g. w^*, x_i) -> keep readable, drop backslash only
    t = re.sub(r'\^([a-zA-Z0-9*]+)', r'^\1', t)
    t = re.sub(r'_([a-zA-Z0-9]+)', r'_\1', t)

    # drop remaining unknown backslash-commands (keep the word)
    t = re.sub(r'\\([a-zA-Z]+)', r'\1', t)

    # collapse now-redundant simple braces left over from stripped commands
    for _ in range(3):
        t = re.sub(r'\{([^{}]*)\}', r'\1', t)

    t = t.replace('\u0001', '{').replace('\u0002', '}')
    t = re.sub(r'[ \t]+', ' ', t)
    t = t.strip()
    return t

def process(md_text):
    out = []
    i = 0
    n = len(md_text)
    while i < n:
        if md_text[i:i+2] == '$$':
            j = md_text.find('$$', i+2)
            if j == -1:
                out.append(md_text[i:])
                break
            formula = md_text[i+2:j]
            converted = convert_math(formula)
            if '\n' in converted or len(converted) > 70:
                out.append('\n```\n' + converted + '\n```\n')
            else:
                out.append('\n> **`' + converted + '`**\n')
            i = j + 2
        elif md_text[i] == '$':
            j = md_text.find('$', i+1)
            if j == -1:
                out.append(md_text[i:])
                break
            formula = md_text[i+1:j]
            if formula.strip() == '':
                out.append('$$')
                i = j + 1
                continue
            converted = convert_math(formula)
            out.append('`' + converted + '`')
            i = j + 1
        else:
            out.append(md_text[i])
            i += 1
    return ''.join(out)

if __name__ == '__main__':
    path = sys.argv[1]
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    code_blocks = []
    def stash(m):
        code_blocks.append(m.group(0))
        return f'@@CODEBLOCK{len(code_blocks)-1}@@'
    protected = re.sub(r'```.*?```', stash, content, flags=re.S)
    converted = process(protected)
    for idx, block in enumerate(code_blocks):
        converted = converted.replace(f'@@CODEBLOCK{idx}@@', block)
    # collapse 3+ consecutive blank lines down to 1 blank line
    converted = re.sub(r'\n{3,}', '\n\n', converted)
    out_path = path.replace('.md', '.converted.md')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(converted)
    print('Wrote', out_path)
