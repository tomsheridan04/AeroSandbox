"""
Generate a pyXDSM data-flow map for CTOL_scoring.ipynb.

This script intentionally does not import or modify CTOL_scoring.ipynb. It
captures the audited data flow as a standalone XDSM artifact.
"""

from pathlib import Path
import shutil
import subprocess

from pyxdsm.XDSM import FUNC, LEFT, METAMODEL, OPT, XDSM


HERE = Path(__file__).resolve().parent
OUTDIR = HERE / "xdsm"
OUTDIR.mkdir(exist_ok=True)


def txt(*lines):
    """Return a pyXDSM multiline text label."""
    escaped_lines = []
    for line in lines:
        escaped = (
            line.replace("\\", r"\textbackslash{}")
            .replace("_", r"\_")
            .replace("%", r"\%")
            .replace("&", r"\&")
        )
        escaped_lines.append(r"\text{" + escaped + "}")
    return tuple(escaped_lines)


def build_pdf_with_available_latex(stem):
    """Compile the pyXDSM TeX source using pdflatex or local Tectonic."""
    if shutil.which("pdflatex") is not None:
        return True

    tectonic = OUTDIR / "tools" / "tectonic-0.16.9" / "tectonic.exe"
    if not tectonic.exists():
        return False

    subprocess.run(
        [
            str(tectonic),
            "--keep-logs",
            "--outdir",
            str(OUTDIR),
            str(OUTDIR / f"{stem}.tex"),
        ],
        check=True,
    )
    return True


def main():
    x = XDSM(use_sfmath=True)

    # External inputs.
    x.add_input(
        "propfit",
        txt("PER3_9x45E.dat", "APC prop table"),
    )
    x.add_input(
        "opt",
        txt("bounds, guesses", "fixed mission constants"),
    )

    # Diagonal systems.
    x.add_system("opt", OPT, txt("D1 Optimizer", "design variables"))
    x.add_system("propfit", METAMODEL, txt("D0 APC", "surrogate fit"))
    x.add_system("geom", FUNC, txt("D2 Geometry", "builder"))
    x.add_system("stab", FUNC, txt("D3 Stability", "VLM"))
    x.add_system("cruise", FUNC, txt("D4 Cruise", "lifting line"))
    x.add_system("stall", FUNC, txt("D5 Stall", "lifting line"))
    x.add_system("turn", FUNC, txt("D6 Turn", "lifting line"))
    x.add_system("climb", FUNC, txt("D7 Climb", "lifting line"))
    x.add_system("vmax", FUNC, txt("D8 Vmax", "lifting line"))
    x.add_system("drag", FUNC, txt("D9 Drag", "buildup"))
    x.add_system("prop", FUNC, txt("D10 Prop/Motor", "coupling"))
    x.add_system("battery", FUNC, txt("D11 Mission", "battery"))
    x.add_system("struct", FUNC, txt("D12 Structures", "spar and boom"))
    x.add_system("weights", FUNC, txt("D13 Weight", "model"))
    x.add_system("score", FUNC, txt("D14 Scoring", "objective"))

    # Design variables and geometry propagation.
    x.connect(
        "opt",
        "geom",
        (
            r"S, AR, \lambda",
            r"S_t, x_t, S_v",
            r"v_s, d_{spar}, t_{tb}",
        ),
    )
    x.connect(
        "geom",
        "stab",
        (r"\text{stability airplane}", r"c_{ref}, v_{cruise}"),
    )
    x.connect(
        "stab",
        "opt",
        (r"x_{np}, C_{m_\alpha}, C_{L_\alpha}", r"\text{SM constraint}"),
    )

    # Flight-condition model inputs.
    flight_inputs = (
        r"\text{airplane geometry}",
        r"v, c_r, c_t, b",
        r"c_{rt}, c_{tt}, b_t",
    )
    for node in ["cruise", "stall", "turn", "climb", "vmax"]:
        x.connect("geom", node, flight_inputs)

    x.connect(
        "opt",
        "cruise",
        (r"\alpha_c, \delta_{e,c}", r"v_{cruise}"),
    )
    x.connect(
        "opt",
        "stall",
        (r"\alpha_s, \delta_{e,s}", r"v_{stall}"),
    )
    x.connect(
        "opt",
        "turn",
        (r"\alpha_t, \delta_{e,t}", r"\phi_{turn}"),
    )
    x.connect(
        "opt",
        "climb",
        (r"\alpha_{cl}, \delta_{e,cl}", r"\gamma_{climb}"),
    )
    x.connect(
        "opt",
        "vmax",
        (r"\alpha_{max}, \delta_{e,max}", r"v_{max}"),
    )

    # Trim and section-load feedback to optimizer/structures.
    for node, suffix in [
        ("cruise", "c"),
        ("stall", "s"),
        ("turn", "t"),
        ("climb", "cl"),
        ("vmax", "max"),
    ]:
        x.connect(
            node,
            "opt",
            (
                rf"L_{suffix}, C_{{m,{suffix}}}",
                rf"\text{{trim residuals}}",
            ),
        )
        x.connect(
            node,
            "drag",
            (
                rf"C_{{D,{suffix}}}",
                rf"v_{suffix}, S, l_f",
            ),
        )

    x.connect(
        "turn",
        "struct",
        (r"y_{forces}", r"L_{forces}", r"\text{turn wing loads}"),
    )
    x.connect(
        "geom",
        "struct",
        (
            r"d_{spar}, t_{tb}",
            r"l_t, S_t, v_{max}",
            r"\text{boom dimensions}",
        ),
    )
    x.connect(
        "struct",
        "opt",
        (
            r"u_{tip}, u'_{tip}",
            r"\sigma_{spar}, \epsilon_{tb}",
            r"\text{structural constraints}",
        ),
    )

    # Propulsion coupling.
    x.connect(
        "propfit",
        "prop",
        (
            r"\hat{T}_{prop}(v,RPM)",
            r"\hat{Q}_{prop}(v,RPM)",
            r"\hat{P}_{prop}(v,RPM)",
        ),
    )
    x.connect(
        "opt",
        "prop",
        (
            r"RPM_i, \tau_i",
            r"W_{prop,max}",
        ),
    )
    x.connect(
        "drag",
        "prop",
        (
            r"D_i",
            r"P_{aero,i}",
            r"\text{required thrust}",
        ),
    )
    x.connect(
        "prop",
        "opt",
        (
            r"T_{prop,i}-D_i",
            r"P_{prop,i}-P_{shaft,i}",
            r"\eta_i",
        ),
    )
    x.connect(
        "prop",
        "battery",
        (
            r"P_{cruise}",
            r"P_{turn}",
            r"P_{climb}",
        ),
    )

    # Battery, weights, and scoring.
    x.connect(
        "battery",
        "weights",
        (
            r"C_{req}",
            r"m_{batt}",
            r"W_{batt}",
        ),
    )
    x.connect(
        "geom",
        "weights",
        (
            r"S, S_t, S_v, b",
            r"l_t, x_{fuse}",
            r"d_{spar}, t_{tb}",
        ),
    )
    x.connect(
        "opt",
        "weights",
        r"W_{prop,max}",
    )
    x.connect(
        "weights",
        "score",
        (r"W_{total}", r"W_{payload}"),
    )
    x.connect(
        "geom",
        "score",
        (r"v_{min}", r"v_{cruise}", r"v_{max}"),
    )
    x.connect(
        "score",
        "opt",
        (
            r"J=\text{flight score}",
            r"\text{takeoff, delivery, RTB, bonus}",
        ),
    )

    # User-facing outputs.
    x.add_output(
        "score",
        (
            r"J^*",
            r"W_{payload}",
            r"\text{score components}",
        ),
        side=LEFT,
    )
    x.add_output(
        "geom",
        (
            r"\text{optimized geometry}",
            r"\text{speeds}",
            r"\text{CG and SM}",
        ),
        side=LEFT,
    )
    x.add_output(
        "prop",
        (
            r"RPM_i, \tau_i",
            r"T_i, P_i, \eta_i",
        ),
        side=LEFT,
    )
    x.add_output(
        "struct",
        (
            r"\text{spar margins}",
            r"\text{tail boom margins}",
        ),
        side=LEFT,
    )

    x.add_process(
        [
            "opt",
            "geom",
            "stab",
            "cruise",
            "stall",
            "turn",
            "climb",
            "vmax",
            "drag",
            "prop",
            "battery",
            "weights",
            "score",
            "opt",
        ],
        arrow=True,
    )

    stem = "ctol_scoring_xdsm"
    use_pdflatex = shutil.which("pdflatex") is not None
    x.write(stem, build=use_pdflatex, cleanup=True, quiet=False, outdir=str(OUTDIR))

    if not use_pdflatex and not build_pdf_with_available_latex(stem):
        raise RuntimeError(
            "No LaTeX compiler found. Install pdflatex, or place tectonic.exe at "
            f"{OUTDIR / 'tools' / 'tectonic-0.16.9' / 'tectonic.exe'}."
        )

    print(f"Wrote pyXDSM sources to: {OUTDIR}")
    print(f"Generated: {stem}.tikz")
    print(f"Generated: {stem}.tex")
    print(f"Generated: {stem}.pdf")


if __name__ == "__main__":
    main()
