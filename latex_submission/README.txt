LaTeX submission package - Medical Image Analysis
================================================

Main file : manuscript-cas.tex
Template  : Elsevier CAS bundle, single-column class (cas-sc.cls)
            This is the LaTeX template recommended in the MIA Guide for Authors.

Contents
--------
  manuscript-cas.tex        main manuscript source
  references.bib            BibTeX database (50 entries)
  manuscript-cas.bbl        formatted bibliography produced by BibTeX
  manuscript-cas.pdf        compiled output of this source (20 pages, A4 preprint)
  cas-sc.cls                Elsevier CAS document class (single column)
  cas-common.sty            CAS shared style file
  cas-model2-names.bst      CAS bibliography style (author-year / Harvard)
  figures/Fig1.pdf ... Fig5.pdf      vector figures (main text)
  figures/Graphical_Abstract.pdf     graphical abstract (vector)
  highlights.tex            research highlights as a LaTeX source
  highlights.txt            research highlights as plain text (5 bullets, <=85 chars each)
  references.md             human-readable reference list (author-year)

How to compile
--------------
Use TeX Live, MiKTeX, or Overleaf (upload this folder and set manuscript-cas.tex
as the main document), then run:

    pdflatex manuscript-cas
    bibtex   manuscript-cas
    pdflatex manuscript-cas
    pdflatex manuscript-cas

The .cls, .sty and .bst files are included so that the project compiles without
installing the CAS bundle separately. If your distribution already provides the
Elsevier CAS bundle, the local copies in this folder take precedence and are
harmless.

Notes
-----
* The package compiles with zero errors and zero undefined citations or
  cross-references using TeX Live 2026 (pdfTeX 3.141592653-2.6-1.40.29).
  One overfull box of 117.08 pt is reported at \maketitle; the identical box
  appears when the unmodified Elsevier cas-sc-sample.tex is compiled, so this
  is a property of the CAS class and not of this manuscript.
* Running heads are typeset above the body text and the running foot reads
  "WANG and LI: Preprint submitted to Elsevier"; page numbers restart after the
  standalone Highlights page, which is the CAS class convention.
* Figure and table captions carry no hard-coded "Figure N." / "Table N." prefix,
  because cas-sc generates those labels itself.
* 5 figures are included as vector PDFs (line art; raster fallback is not needed).
* 7 tables are typeset in the manuscript with booktabs rules; there are no
  vertical rules and no cell shading, in line with the journal's table rules.
* References use author-year citations (\citep) with cas-model2-names.bst.
* Long author lists are never expanded in citations: natbib is loaded without
  the 'longnamesfirst' option, so a consensus statement with 40 authors is
  cited as "Wynants et al., 2020".
