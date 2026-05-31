#!/usr/bin/env python3
"""
oMath builder module for Word equation objects.
Used by gen_jmp_docx_v2.py to create proper Word equations instead of plain text.
"""
from lxml import etree
import re

M_NS = 'http://schemas.openxmlformats.org/officeDocument/2006/math'
W_NS = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'

def m(tag): return f'{{{M_NS}}}{tag}'
def w(tag): return f'{{{W_NS}}}{tag}'

def make_mr(text, italic=True):
    r = etree.Element(m('r'))
    rPr = etree.SubElement(r, m('rPr'))
    if italic:
        sty = etree.SubElement(rPr, m('sty'))
        sty.set(m('val'), 'i')
    t = etree.SubElement(r, m('t'))
    t.text = text
    return r

def make_sub(base_text, sub_text, base_italic=True):
    sSub = etree.Element(m('sSub'))
    e = etree.SubElement(sSub, m('e'))
    e.append(make_mr(base_text, italic=base_italic))
    sub = etree.SubElement(sSub, m('sub'))
    sub.append(make_mr(sub_text, italic=False))
    return sSub

def make_sup_on_element(base_element, sup_text):
    sSup = etree.Element(m('sSup'))
    e = etree.SubElement(sSup, m('e'))
    e.append(base_element)
    sup = etree.SubElement(sSup, m('sup'))
    sup.append(make_mr(sup_text, italic=False))
    return sSup

def make_frac(num_children, den_children):
    f = etree.Element(m('f'))
    fPr = etree.SubElement(f, m('fPr'))
    ctrlPr = etree.SubElement(fPr, m('ctrlPr'))
    num = etree.SubElement(f, m('num'))
    for child in num_children: num.append(child)
    den = etree.SubElement(f, m('den'))
    for child in den_children: den.append(child)
    return f

def make_hat(base_text, sub_text=None):
    acc = etree.Element(m('acc'))
    accPr = etree.SubElement(acc, m('accPr'))
    chr_elem = etree.SubElement(accPr, m('chr'))
    chr_elem.set(m('val'), '\u0302')
    ctrlPr = etree.SubElement(accPr, m('ctrlPr'))
    e = etree.SubElement(acc, m('e'))
    if sub_text:
        e.append(make_sub(base_text, sub_text))
    else:
        e.append(make_mr(base_text))
    return acc

def make_d(children):
    d = etree.Element(m('d'))
    dPr = etree.SubElement(d, m('dPr'))
    ctrlPr = etree.SubElement(dPr, m('ctrlPr'))
    e = etree.SubElement(d, m('e'))
    for child in children: e.append(child)
    return d

def make_func(name_children, arg_children):
    func = etree.Element(m('func'))
    funcPr = etree.SubElement(func, m('funcPr'))
    ctrlPr = etree.SubElement(funcPr, m('ctrlPr'))
    fName = etree.SubElement(func, m('fName'))
    for child in name_children: fName.append(child)
    e = etree.SubElement(func, m('e'))
    for child in arg_children: e.append(child)
    return func

def make_omath_para(omath):
    omathPara = etree.Element(m('oMathPara'))
    omathParaPr = etree.SubElement(omathPara, m('oMathParaPr'))
    jc = etree.SubElement(omathParaPr, m('jc'))
    jc.set(m('val'), 'center')
    omathPara.append(omath)
    return omathPara

def add_eq_number(para_element, eq_num):
    r = etree.SubElement(para_element, w('r'))
    rPr = etree.SubElement(r, w('rPr'))
    rFonts = etree.SubElement(rPr, w('rFonts'))
    rFonts.set(w('ascii'), 'Times New Roman')
    rFonts.set(w('hAnsi'), 'Times New Roman')
    t = etree.SubElement(r, w('t'))
    t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
    t.text = f'  ({eq_num})'


# --- Pre-built formulas ---

def build_lm_score():
    omath = etree.Element(m('oMath'))
    omath.append(make_sub('LM', 't'))
    omath.append(make_mr(' = ', italic=False))
    omath.append(make_frac(
        [make_mr('Positive words\u209c \u2212 Negative words\u209c', italic=False)],
        [make_mr('Total words\u209c', italic=False)]
    ))
    return omath

def build_cb_score():
    omath = etree.Element(m('oMath'))
    omath.append(make_sub('CB', 't'))
    omath.append(make_mr(' = ', italic=False))
    omath.append(make_frac(
        [make_mr('Hawkish words\u209c \u2212 Dovish words\u209c', italic=False)],
        [make_mr('Total words\u209c', italic=False)]
    ))
    return omath

def build_combined():
    omath = etree.Element(m('oMath'))
    omath.append(make_sub('S', 't'))
    omath.append(make_mr(' = 0.5 \u00d7 ', italic=False))
    omath.append(make_sub('LM', 't'))
    omath.append(make_mr(' + 0.5 \u00d7 ', italic=False))
    omath.append(make_sub('CB', 't'))
    return omath

def build_h1_regression():
    omath = etree.Element(m('oMath'))
    omath.append(make_sub('S', 't'))
    omath.append(make_mr(' = ', italic=False))
    omath.append(make_mr('\u03b1'))
    omath.append(make_mr(' + ', italic=False))
    omath.append(make_sub('\u03b2', '1'))
    omath.append(make_mr(' \u00b7 ', italic=False))
    omath.append(make_sub('Target', 't'))
    omath.append(make_mr(' + ', italic=False))
    omath.append(make_sub('\u03b2', '2'))
    omath.append(make_mr(' \u00b7 ', italic=False))
    omath.append(make_sub('Path', 't'))
    omath.append(make_mr(' + ', italic=False))
    omath.append(make_sub('\u03b5', 't'))
    return omath

def build_h2_regression():
    omath = etree.Element(m('oMath'))
    omath.append(make_sub('R', 't'))
    omath.append(make_mr(' = ', italic=False))
    omath.append(make_mr('\u03b1'))
    omath.append(make_mr(' + ', italic=False))
    omath.append(make_sub('\u03b2', '1'))
    omath.append(make_mr(' \u00b7 ', italic=False))
    omath.append(make_sub('Target', 't'))
    omath.append(make_mr(' + ', italic=False))
    omath.append(make_sub('\u03b2', '2'))
    omath.append(make_mr(' \u00b7 ', italic=False))
    omath.append(make_sub('Path', 't'))
    omath.append(make_mr(' + ', italic=False))
    omath.append(make_sub('\u03b5', 't'))
    return omath

def build_wald():
    omath = etree.Element(m('oMath'))
    omath.append(make_mr('W'))
    omath.append(make_mr(' = ', italic=False))
    paren = make_d([
        make_hat('\u03b2', '1'),
        make_mr(' \u2212 ', italic=False),
        make_hat('\u03b2', '2'),
    ])
    omath.append(make_frac(
        [make_sup_on_element(paren, '2')],
        [make_func(
            [make_mr('Var', italic=False)],
            [make_hat('\u03b2', '1'), make_mr(' \u2212 ', italic=False), make_hat('\u03b2', '2')]
        )]
    ))
    return omath

def build_h3_regression():
    omath = etree.Element(m('oMath'))
    omath.append(make_sub('R', 't'))
    omath.append(make_mr(' = ', italic=False))
    omath.append(make_mr('\u03b1'))
    omath.append(make_mr(' + ', italic=False))
    omath.append(make_sub('\u03b2', '1'))
    omath.append(make_mr(' \u00b7 ', italic=False))
    omath.append(make_sub('Target', 't'))
    omath.append(make_mr(' + ', italic=False))
    omath.append(make_sub('\u03b2', '2'))
    omath.append(make_mr(' \u00b7 ', italic=False))
    omath.append(make_sub('Path', 't'))
    omath.append(make_mr(' + ', italic=False))
    omath.append(make_sub('\u03b2', '3'))
    omath.append(make_mr(' \u00b7 ', italic=False))
    omath.append(make_sub('S', 't'))
    omath.append(make_mr(' + ', italic=False))
    omath.append(make_sub('\u03b2', '4'))
    omath.append(make_mr(' \u00b7 ', italic=False))
    omath.append(make_d([
        make_sub('S', 't'),
        make_mr(' \u00d7 ', italic=False),
        make_sub('FG', 't'),
    ]))
    omath.append(make_mr(' + ', italic=False))
    omath.append(make_sub('\u03b5', 't'))
    return omath

def build_dynamic():
    omath = etree.Element(m('oMath'))
    omath.append(make_sub('S', 't'))
    omath.append(make_mr(' = ', italic=False))
    omath.append(make_mr('\u03b1'))
    omath.append(make_mr(' + ', italic=False))
    omath.append(make_mr('\u03c1'))
    omath.append(make_sub('S', 't\u22121'))
    omath.append(make_mr(' + ', italic=False))
    omath.append(make_sub('\u03b2', '1'))
    omath.append(make_mr(' \u00b7 ', italic=False))
    omath.append(make_sub('Target', 't'))
    omath.append(make_mr(' + ', italic=False))
    omath.append(make_sub('\u03b2', '2'))
    omath.append(make_mr(' \u00b7 ', italic=False))
    omath.append(make_sub('Path', 't'))
    omath.append(make_mr(' + ', italic=False))
    omath.append(make_sub('\u03b5', 't'))
    return omath


# --- Formula registry ---

FORMULA_REGISTRY = [
    # Order matters: more specific patterns first
    # LaTeX patterns (from markdown with backslash-alpha etc.)
    (r'^W\s*=.*\\frac', build_wald, '6'),
    (r'^R_t\s*=.*\\beta_3', build_h3_regression, '7'),
    (r'^S_t\s*=.*\\rho', build_dynamic, '8'),
    (r'^S_t\s*=.*\\alpha.*Target', build_h1_regression, '4'),
    (r'^R_t\s*=.*\\alpha.*Target', build_h2_regression, '5'),
    # Unicode patterns (from clean_latex output or broken docx)
    (r'^W\s*=.*\u0302.*Var', build_wald, '6'),
    (r'^R[\u209c_]\s*=.*\u03b2[\u2083\u2084]', build_h3_regression, '7'),
    (r'^S[\u209c_]\s*=.*\u03c1.*S', build_dynamic, '8'),
    (r'^S[\u209c_]\s*=.*\u03b1.*Target', build_h1_regression, '4'),
    (r'^R[\u209c_]\s*=.*\u03b1.*Target', build_h2_regression, '5'),
    (r'^LM[\u209c_]\s*=.*Positive', build_lm_score, '1'),
    (r'^CB[\u209c_]\s*=.*Hawkish', build_cb_score, '2'),
    (r'^S[\u209c_]\s*=\s*0\.5', build_combined, '3'),
]

def match_formula(latex_text):
    """Try to match a LaTeX formula text to a pre-built oMath formula.
    Returns (builder_func, eq_number) or None."""
    for pattern, builder, eq_num in FORMULA_REGISTRY:
        if re.search(pattern, latex_text):
            return builder, eq_num
    return None
