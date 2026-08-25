"""Class names for the element classifier.

Order MUST match the trained model's output indices, which was derived from
sorted(dataset_final) subdirectory names in training.ipynb. The trailing
``_rN`` encodes the element's in-plane rotation (r0..r3) as seen in the photo;
Element.getCircuiTikZLabel relies on it to decide whether the CircuiTikZ
label needs ",invert".
"""

CLASSES = [
    "ac_src_r0",
    "ac_src_r1",
    "battery_r0",
    "battery_r1",
    "battery_r2",
    "battery_r3",
    "cap_r0",
    "cap_r1",
    "curr_src_r0",
    "curr_src_r1",
    "curr_src_r2",
    "curr_src_r3",
    "dc_volt_src_1_r0",
    "dc_volt_src_1_r1",
    "dc_volt_src_1_r2",
    "dc_volt_src_1_r3",
    "dc_volt_src_2_r0",
    "dc_volt_src_2_r1",
    "dc_volt_src_2_r2",
    "dc_volt_src_2_r3",
    "dep_curr_src_r0",
    "dep_curr_src_r1",
    "dep_curr_src_r2",
    "dep_curr_src_r3",
    "dep_volt_r0",
    "dep_volt_r1",
    "dep_volt_r2",
    "dep_volt_r3",
    "diode_r0",
    "diode_r1",
    "diode_r2",
    "diode_r3",
    "gnd_1",
    "inductor_r0",
    "inductor_r1",
    "resistor_r0",
    "resistor_r1",
]
assert len(CLASSES) == 37
