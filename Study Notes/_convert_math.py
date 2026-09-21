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

def repl_frac(text):
    # repeatedly replace innermost \frac{a}{b} or \dfrac/\tfrac
    pattern = re.compile(r'\\[dt]?frac\{([^{}]*)\}\{([^{}]*)\}')
    prev = None
    while prev != text:
        prev = text
        text = pattern.sub(lambda m: f'({m.group(1)})/({m.group(2)})', text)
    return text

def repl_sqrt(text):
    pattern = re.compile(r'\\sqrt\{([^{}]*)\}')
    prev = None
    while prev != text:
        prev = text
        text = pattern.sub(lambda m: f'\u221a({m.group(1)})', text)
    # \sqrt without braces followed by single token
    text = re.sub(r'\\sqrt\b', '\u221a', text)
    return text

def superscriptify(m):
    body = m.group(1)
    if all(c in SUP_DIGITS for c in body):
        return ''.join(SUP_DIGITS[c] for c in body)
    return '^' + body

def subscriptify(m):
    body = m.group(1)
    if all(c in SUB_DIGITS for c in body):
        return ''.join(SUB_DIGITS[c] for c in body)
    return '_' + body

def convert_math(text):
    t = text

    # braces-wrapped commands first
    t = repl_frac(t)
    t = repl_sqrt(t)

    # \left \right and big variants -> drop, keep the delimiter char that follows
    t = re.sub(r'\\[bB]igg?[lr]?\(', '(', t)
    t = re.sub(r'\\[bB]igg?[lr]?\)', ')', t)
    t = re.sub(r'\\left\(', '(', t)
    t = re.sub(r'\\right\)', ')', t)
    t = re.sub(r'\\left\[', '[', t)
    t = re.sub(r'\\right\]', ']', t)
    t = re.sub(r'\\left\\\{', '{', t)
    t = re.sub(r'\\right\\\}', '}', t)
    t = re.sub(r'\\left\\lfloor', '\u230a', t)
    t = re.sub(r'\\right\\rfloor', '\u230b', t)
    t = re.sub(r'\\lfloor', '\u230a', t)
    t = re.sub(r'\\rfloor', '\u230b', t)
    t = re.sub(r'\\left\\lceil', '\u2308', t)
    t = re.sub(r'\\right\\rceil', '\u2309', t)
    t = re.sub(r'\\left\\\|', '\u2016', t)
    t = re.sub(r'\\right\\\|', '\u2016', t)
    t = re.sub(r'\\left\|', '|', t)
    t = re.sub(r'\\right\|', '|', t)
    t = re.sub(r'\\left\.', '', t)
    t = re.sub(r'\\right\.', '', t)
    t = re.sub(r'\\left', '', t)
    t = re.sub(r'\\right', '', t)

    # boxed{X} -> [X]
    t = re.sub(r'\\boxed\{([^{}]*)\}', r'[\1]', t)

    # text{...}
    t = re.sub(r'\\text\{([^{}]*)\}', r'\1', t)
    t = re.sub(r'\\mathrm\{([^{}]*)\}', r'\1', t)
    t = re.sub(r'\\mathbf\{([^{}]*)\}', r'\1', t)
    t = re.sub(r'\\mathbb\{R\}', '\u211d', t)
    t = re.sub(r'\\mathcal\{N\}', '\U0001D4A9', t)
    t = re.sub(r'\\mathcal\{B\}', '\U0001D4B7', t)
    t = re.sub(r'\\mathcal\{L\}', 'L', t)

    # hat / bar accents for single-letter args
    t = re.sub(r"\\hat\{([a-zA-Z])\}", lambda m: m.group(1) + '\u0302', t)
    t = re.sub(r"\\bar\{([a-zA-Z])\}", lambda m: m.group(1) + '\u0304', t)
    t = re.sub(r"\\tilde\{([a-zA-Z])\}", lambda m: m.group(1) + '\u0303', t)
    t = re.sub(r"\\hat ([a-zA-Z])", lambda m: m.group(1) + '\u0302', t)

    # sum / prod with limits: \sum_{i=1}^{n} -> "sum(i=1..n)"
    t = re.sub(r'\\sum_\{([^{}]*)\}\^\{([^{}]*)\}', r'Σ(\1..\2)', t)
    t = re.sub(r'\\sum_\{([^{}]*)\}', r'Σ(\1)', t)
    t = re.sub(r'\\sum_([a-zA-Z0-9])\^([a-zA-Z0-9])', r'Σ(\1..\2)', t)
    t = re.sub(r'\\sum_([a-zA-Z0-9])', r'Σ(\1)', t)
    t = re.sub(r'\\sum\b', 'Σ', t)

    t = re.sub(r'\\prod_\{([^{}]*)\}\^\{([^{}]*)\}', r'Π(\1..\2)', t)
    t = re.sub(r'\\prod_\{([^{}]*)\}', r'Π(\1)', t)
    t = re.sub(r'\\prod\b', 'Π', t)

    t = re.sub(r'\\int_\{([^{}]*)\}\^\{([^{}]*)\}', r'∫(\1..\2)', t)
    t = re.sub(r'\\int\b', '∫', t)

    t = re.sub(r'\\lim_\{([^{}]*)\}', r'lim(\1)', t)

    # partial, nabla, infty
    t = t.replace('\\partial', '\u2202')
    t = t.replace('\\nabla', '\u2207')
    t = t.replace('\\infty', '\u221e')
    t = t.replace('\\top', '\u1d40')
    t = t.replace('\\cdot', '\u00b7')
    t = t.replace('\\times', '\u00d7')
    t = t.replace('\\pm', '\u00b1')
    t = t.replace('\\approx', '\u2248')
    t = t.replace('\\neq', '\u2260')
    t = t.replace('\\leq', '\u2264')
    t = t.replace('\\geq', '\u2265')
    t = t.replace('\\ll', '\u226a')
    t = t.replace('\\gg', '\u226b')
    t = t.replace('\\in', '\u2208')
    t = t.replace('\\notin', '\u2209')
    t = t.replace('\\subset', '\u2282')
    t = t.replace('\\forall', '\u2200')
    t = t.replace('\\exists', '\u2203')
    t = t.replace('\\Rightarrow', '\u21d2')
    t = t.replace('\\Leftrightarrow', '\u21d4')
    t = t.replace('\\rightarrow', '\u2192')
    t = t.replace('\\to', '\u2192')
    t = t.replace('\\leftarrow', '\u2190')
    t = t.replace('\\Leftarrow', '\u21d0')
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

    # arg min / arg max
    t = re.sub(r'\\arg\\?min_?\{?([^{}\s\\]*)\}?', r'argmin(\1)', t)
    t = re.sub(r'\\arg\\?max_?\{?([^{}\s\\]*)\}?', r'argmax(\1)', t)
    t = t.replace('\\min', 'min')
    t = t.replace('\\max', 'max')
    t = t.replace('\\log', 'log')
    t = t.replace('\\exp', 'exp')
    t = t.replace('\\sign', 'sign')

    # Greek letters: \name -> unicode  (word-boundary to avoid partial matches like \lambda vs \lambdaX)
    for name, sym in sorted(GREEK.items(), key=lambda kv: -len(kv[0])):
        t = re.sub(r'\\' + name + r'\b', sym, t)

    # remaining \mathcal{X} generic -> X
    t = re.sub(r'\\mathcal\{([^{}]*)\}', r'\1', t)

    # superscripts / subscripts with braces
    t = re.sub(r'\^\{([^{}]*)\}', superscriptify, t)
    t = re.sub(r'_\{([^{}]*)\}', subscriptify, t)
    # single-char superscripts/subscripts without braces
    t = re.sub(r'\^([0-9+\-])', superscriptify, t)
    t = re.sub(r'_([0-9+\-])', subscriptify, t)
    # remaining single-letter/number sub without braces (e.g., w_i, x_1 already handled by digit rule; letters keep _)
    # leave \^ and _ followed by letters as-is (e.g. w_i -> keep w_i, readable enough)

    # clean any leftover backslash-command residues (unknown commands) - drop backslash but keep name
    t = re.sub(r'\\([a-zA-Z]+)', r'\1', t)

    # collapse now-redundant braces left from stripped commands: {x} -> x  (only if content has no braces)
    t = re.sub(r'\{([^{}]*)\}', r'\1', t)
    t = re.sub(r'\{([^{}]*)\}', r'\1', t)

    # tidy extra spaces
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
            out.append('\n```\n' + converted + '\n```\n')
            i = j + 2
        elif md_text[i] == '$':
            # avoid matching inside code fences; simple approach - find next $
            j = md_text.find('$', i+1)
            if j == -1:
                out.append(md_text[i:])
                break
            formula = md_text[i+1:j]
            # skip if this looks like a currency amount (unlikely in this doc) or empty
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
    # protect fenced code blocks (```...```) from $ processing
    code_blocks = []
    def stash(m):
        code_blocks.append(m.group(0))
        return f'@@CODEBLOCK{len(code_blocks)-1}@@'
    protected = re.sub(r'```.*?```', stash, content, flags=re.S)
    converted = process(protected)
    for idx, block in enumerate(code_blocks):
        converted = converted.replace(f'@@CODEBLOCK{idx}@@', block)
    out_path = path.replace('.md', '.converted.md')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(converted)
    print('Wrote', out_path)
