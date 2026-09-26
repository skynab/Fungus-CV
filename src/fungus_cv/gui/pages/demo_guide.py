"""How to walk through the two demo experiments, with buttons to make them."""

from __future__ import annotations

from PySide6.QtWidgets import QHBoxLayout, QPushButton, QTextBrowser, QVBoxLayout, QWidget

from fungus_cv.gui import theme

# Links of the form page:<title> open that page of the app.
GUIDE = """
<p>Two synthetic experiments let you try the whole program without a camera. They follow the
two ways of telling it what to measure: <b>colour thresholds</b> (Set Up Measurement) and
<b>SAM 2 clicks</b> (SAM Prompts). The truth behind every photo is known, so you can check what
the program finds. Each demo is made in a new folder you choose and opens straight away.</p>

<h2>1. Dye strip &mdash; the Set Up Measurement workflow</h2>
<p>A paper towel strip wicking blue dye, photographed every 2 minutes for an hour. The dye
rises as h = 12 mm &middot; &radic;(t &minus; 0.5 min). Everything is set up already: it is
ready to analyse.</p>
<ol>
<li><b>Make it</b> with the button above (or File &rarr; Make Demo Experiment&hellip;).</li>
<li>Optional: open <a href="page:Set Up Measurement">Set Up Measurement</a> to see what was
set up for you: the base and tip of the strip, the region around it, and the blue colour
being followed. For your own experiments, this is where you would click them.</li>
<li>Open <a href="page:Analyze">Analyze</a> and press <b>Run</b>. Every photo is measured
(under a minute). Browse the frames to see the mask on the dye.</li>
<li>Open <a href="page:Report">Report</a>, leave Measurement on <b>Extent</b> and press
<b>Make report</b>. The <i>power</i> model should fit best with n close to 0.5, a square
root, and the <i>sqrt</i> model's k should be close to 12.</li>
<li>Still on Report, try the other tabs. <b>Spread map</b> and <b>Sensitivity check</b> each
have their own button. For <b>Conditions</b>, choose <i>temperature_c</i> under Conditions
and press Make report again: the room temperature is drawn under the measurement.</li>
<li>Open <a href="page:Validation">Validation</a> and choose <i>hand_measurements.csv</i> from
the experiment folder to compare the program with a person's ruler readings. The folder's
<i>suite.yaml</i> is a ready-made validation suite.</li>
</ol>

<h2>2. Mould colony &mdash; the SAM Prompts workflow</h2>
<p>A colony spreading over an agar plate, photographed every 6 hours for six days. After a
12 h lag its radius grows 0.22 mm/h, 1.4&times; faster to the right than to the left. It has
no colour of its own to threshold on, so it is set to SAM 2, and telling SAM 2 what the colony
is, is your part.</p>
<p><b>Needs SAM 2</b> (PyTorch): if <a href="page:Diagnostics">Diagnostics</a> reports it
missing, install it with <code>pip install -e ".[sam]"</code> in the program's folder.
Without a graphics card it runs on the processor, just more slowly.</p>
<ol>
<li><b>Make it</b> with the button above (or File &rarr; Make SAM 2 Demo Experiment&hellip;).
It opens on <a href="page:SAM Prompts">SAM Prompts</a>.</li>
<li>Drag the <b>Frame</b> slider to a frame where the colony is clearly visible (about halfway
through).</li>
<li><b>Left-click the white rim</b>, then <b>left-click the green centre</b>. After each click
SAM 2 shows its answer in magenta; the first click takes a few seconds while the model loads.
Check that the magenta covers the whole colony, rim included: a single click on the centre
gets only the centre. Right-click marks something as <i>not</i> the colony.</li>
<li>Press <b>Save prompt for this frame</b>. SAM 2 follows the colony from there, forwards and
backwards, through the whole time-lapse. Prompting a second frame corrects the tracking from
that frame on, if it ever drifts.</li>
<li>Open <a href="page:Analyze">Analyze</a> and press <b>Run</b>. On the processor this takes
a few minutes.</li>
<li>Open <a href="page:Report">Report</a>, set Measurement to <b>Equivalent radius</b> and
press <b>Make report</b>. The <i>linear</i> model should fit best. Time is in days, so its
slope b should be close to 5 mm/day (0.22 mm/h).</li>
<li>Press <b>Spread map</b>: the colony should grow fastest toward 0&deg; (to the right), with
a fastest/slowest ratio around 1.4. Choose <i>temperature_c</i> under Conditions to see the
room's day and night.</li>
<li>Open <a href="page:Validation">Validation</a> with the folder's
<i>hand_measurements.csv</i> and Measurement <b>Equivalent radius</b>.</li>
</ol>

<h2>What is in a demo folder</h2>
<ul>
<li><i>demo_truth.csv</i>: the true value of every photo (height or radius).</li>
<li><i>hand_measurements.csv</i> and <i>suite.yaml</i>: ruler readings with a realistic
reading error, and a validation suite that uses them.</li>
<li><i>room_logger.csv</i>: a temperature and humidity logger's file, already imported. Import
your own logger's file with <b>Import log&hellip;</b> on the Report page.</li>
</ul>
"""


class DemoGuidePage(QWidget):
    def __init__(self, state, window=None):
        super().__init__()
        self.state = state
        self.window = window
        self.dye_btn = QPushButton("Make the dye strip demo…")
        theme.mark_primary(self.dye_btn)
        self.sam_btn = QPushButton("Make the SAM 2 colony demo…")
        theme.mark_primary(self.sam_btn)
        if window is not None:
            self.dye_btn.clicked.connect(lambda: window.make_demo())
            self.sam_btn.clicked.connect(lambda: window.make_demo(sam=True))
        self.text = QTextBrowser()
        self.text.setOpenLinks(False)
        self.text.anchorClicked.connect(self._link)
        self.text.setHtml(GUIDE)
        buttons = QHBoxLayout()
        buttons.addWidget(self.dye_btn)
        buttons.addWidget(self.sam_btn)
        buttons.addStretch()
        layout = QVBoxLayout(self)
        layout.addLayout(buttons)
        layout.addWidget(self.text, 1)

    def _link(self, url) -> None:
        target = url.toString()
        if target.startswith("page:") and self.window is not None:
            self.window.go_to(target[len("page:"):])
