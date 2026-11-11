# Independent isolated-wing reference

Source: C. H. Zimmerman, **NACA Report 431, Characteristics of Clark Y Airfoils of
Small Aspect Ratios** (1932), [NASA record](https://ntrs.nasa.gov/citations/19930091504),
[original PDF](https://ntrs.nasa.gov/api/citations/19930091504/downloads/19930091504.pdf).
The original public US-government report is included unchanged as `naca-tr-431.pdf`.

`clark_y_published_ordinates.csv` transcribes Table I, printed page 11 / PDF page 13.
Percent-chord values are divided by 100. The historical coordinate reference, including
the leading-edge ordinate, is preserved. AeroSandbox's airfoil normalization is applied
internally when evaluating NeuralFoil; no angle offset is fitted to the measurements.

`clark_y_ar6_yaw_minus20.csv` transcribes Table II on the same page. It is the isolated
rectangular-tip wing at **aspect ratio 6 and -20 degrees yaw**, with total wing CD.
These are finite-wing results, not profile-drag coefficients or 2-D airfoil polars.
The source states that alpha and drag are corrected for tunnel-wall effects.
Only alpha <= 30 degrees is compared in the notebook; the rest of the original table
is retained for traceability. The sampled first peak is CL = 1.194 at alpha = 16.2 degrees.

The operating-point reconstruction uses the report's 80 mph condition and 14.14-inch
reference chord, corresponding to approximately Re = 860,000. Span is 6 chords. The
modern standard sea-level atmosphere is an approximation to the historical test state.
Unknown transition/turbulence and model/manufacturing details remain comparison limits.

Table IX, printed page 12 / PDF page 14, separately gives the **zero-yaw** rectangular
AR-6 wing CLmax = 1.240, alpha at maximum resultant force = 15 degrees, and CDmin = 0.0151.
The notebook compares the zero-yaw maximum lift coefficient against a separate beta=0
sweep. The table's 15-degree angle belongs to the maximum **resultant** force, not
necessarily the precise maximum-lift angle, and is not used as a stall-angle target.

The transcription and code were created with Codex. Human review against the source
pages is outstanding. `sources.json` identifies the source and checksums; notebook run
metadata also records data and implementation hashes.
