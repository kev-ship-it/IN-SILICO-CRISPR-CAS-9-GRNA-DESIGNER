import sys
import os
import streamlit as st
import streamlit.components.v1 as components

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)
from pipeline.design_grna import design_best_grna

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="CRISPR-ML Studio",
    page_icon="🧬",
    layout="wide"
)

# =========================================================
# OFF-TARGET RISK INTERPRETATION (ADDED)
# =========================================================
def off_target_risk_label(score):
    if score < 0.20:
        return "LOW", "🟢", "#00ff99"
    elif score < 0.50:
        return "MEDIUM", "🟡", "#ffd166"
    else:
        return "HIGH", "🔴", "#ff4e50"

def on_target_risk_label(score):
    if score < 0.20:
        return "LOW", "🔴", "#ff4e50"
    elif score < 0.50:
        return "MEDIUM", "🟡", "#ffd166"
    else:
        return "HIGH", "🟢", "#00ff99"

# =========================================================
# CSS – DYNAMIC COLOR GRADIENT BACKGROUND
# =========================================================
st.markdown(
"""
<style>
body {
    background: linear-gradient(135deg, #0b3d0b, #145214, #2a7a2a, #1d4d1d);
    background-size: 400% 400%;
    animation: gradientAnimation 20s ease infinite;
    font-family: 'Segoe UI', sans-serif;
    color: #ffffff;
}

@keyframes gradientAnimation {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

h1,h2,h3 {
    color: #ffffff;
    text-shadow: 0 0 12px rgba(0,0,0,0.6);
    text-align: center;
}

.card {
    background: linear-gradient(135deg, #0d1b8f, #1dd3b0, #6a11cb, #00ff99)  !important;
    border-radius: 18px;
    padding: 24px;
    margin-bottom: 24px;
    border: 1px solid rgba(255,255,255,0.15);
    box-shadow: 0 10px 35px rgba(0,0,0,0.4);
    transition: transform 0.2s, box-shadow 0.2s;
}

.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 15px 45px rgba(0,255,255,0.3);
}

textarea {
    background-color: rgba(0,0,0,0.7) !important;
    color: #ffffff !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.25) !important;
}

button {
    background: linear-gradient(135deg, #00ffff, #ff00ff) !important;
    color: #001018 !important;
    font-weight: 700 !important;
    border-radius: 16px !important;
    padding: 10px 25px !important;
    box-shadow: 0 8px 25px rgba(0,255,255,0.4);
}

button:hover {
    transform: scale(1.06);
}

.metric {
    background: rgba(0,0,0,0.35) !important;
    border-radius: 12px;
    padding: 10px;
}
</style>
""",
unsafe_allow_html=True
)

# =========================================================
# HEADER
# =========================================================
st.markdown(
"""
<div style="text-align:center; margin-bottom:25px;">
    <h1>🧬 CRISPR-ML Studio</h1>
    <p>A ML based project studio to explore the world of CRISPR-cas9 gene editing</p>
</div>
""",
unsafe_allow_html=True
)

# =========================================================
# INPUT
# =========================================================
dna_sequence = st.text_area(
    "Input DNA sequence (5′ → 3′)",
    height=120,
    placeholder="ATGCGTACGATCGATCGATCGATCGATCGATCG"
).upper().strip()

cas9_option = st.selectbox(
    "Select Cas9 Protein",
    [
        "SpCas9 (NGG)",
        "SaCas9 (NNGRRT)",
        "StCas9 (NNAGAAW)"
    ]
)

run_btn = st.button("🚀 Run CRISPR Simulation")
st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# HOW TO USE – USER GUIDANCE + SAMPLE INPUTS
# =========================================================
with st.expander("🧭 How to use this tool (Click to expand)", expanded=True):
   st.markdown(f"""
<div style="
    display:flex;
    justify-content:center;
">
  <div style="
      max-width:700px;
      text-align:center;
      line-height:1.8;
  ">
    <h3>🧬 Step-by-step guide</h3>

    <p>
      <b>1. Paste a DNA sequence</b> (5′ → 3′ direction)<br>
      • Only use <b>A, T, C, G</b><br>
      • Recommended length: <b>≥ 40 nucleotides</b>
    </p>

    <p>
      <b>2. Select a Cas9 variant</b><br>
      • <b>SpCas9</b> → NGG PAM (most common)<br>
      • <b>SaCas9</b> → NNGRRT PAM (compact Cas9)<br>
      • <b>StCas9</b> → NNAGAAW PAM (thermophilic)
    </p>

    <p>
      <b>3.</b> Click <b>🚀 Run CRISPR Simulation</b>
    </p>

    <p>
      <b>4. Interpret results</b><br>
      🧪 <b>On-Target Efficiency</b> → cutting performance<br>
      🧪 <b>Off-Target Risk</b> → unintended binding likelihood<br>
      🎥 <b>3D View</b> → DNA unwinding + gRNA–Cas9 binding
    </p>
  </div>
</div>
""", unsafe_allow_html=True)


    st.markdown("### 📌 Sample DNA sequences (click to copy)")

    st.code(
        "ATGCGTACGGATCGATCGGATCCGATCGGATCGATCGTACGATCG",
        language="text"
    )
    st.caption("✅ Works with **SpCas9 (NGG PAM)**")

    st.code(
        "GCTAGCTAGCTAGGAGGTTACGATCGATCGGATCGATCGATCGATCGA",
        language="text"
    )
    st.caption("✅ Compatible with **SaCas9 (NNGRRT PAM)**")

    st.code(
        "ATCGATCGATAGAAATCGATCGATCGACGAGAATTATCGATCGATCGATCGA",
        language="text"
    )
    st.caption("✅ Compatible with **StCas9 (NNAGAAW PAM)**")

# =========================================================
# RUN PIPELINE
# =========================================================
if run_btn:
    result = design_best_grna(dna_sequence, cas9_type=cas9_option)

    if result is None:
        st.error(f"❌ No valid PAM sites detected for {cas9_option}")
        st.stop()

    risk_label, risk_icon, risk_color = off_target_risk_label(result["off"])
    eff_label, eff_icon, eff_color = on_target_risk_label(result["on"])
    gRNA = result["seq"].replace("T", "U")  # convert T → U for display only
    start = result["start"]
    end = result["end"]

    col1, col2 = st.columns([1, 2])

    # =====================================================
    # LEFT: METRICS
    # =====================================================
    with col1:
        st.markdown(f'<div class="card"><center>Target site: {start} → {end}</center></div>', unsafe_allow_html=True)
        st.subheader("📊 ML Prediction")

        st.markdown(
        f"""
        <div class="card" style="text-align:center;">
            <h3>🧪On-Target Efficiency (%)</h3>
            <h1 style="color:{eff_color}; font-size:42px;">
                {eff_icon} {eff_label}
            </h1>
            <p style="opacity:0.8;">
                Score: {result["on"]*100:.3f}
            </p>
        </div>
        """,
        unsafe_allow_html=True
        )

        st.markdown(
        f"""
        <div class="card" style="text-align:center;">
            <h3>🧪 Off-Target Risk</h3>
            <h1 style="color:{risk_color}; font-size:42px;">
                {risk_icon} {risk_label}
            </h1>
            <p style="opacity:0.8;">
                Score: {result["off"]*100:.3f}
            </p>
        </div>
        """,
        unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # =====================================================
    # RIGHT: 3D MOLECULAR SIMULATION  
    # =====================================================
    with col2:
        st.markdown(f'<div class="card"><center>Cas9 Protein: {cas9_option}</center><br><center>gRNA sequence : {gRNA}</center></div>', unsafe_allow_html=True)
        st.subheader("🎥 Molecular Mechanism Simulation")

        html = f"""
        <div id="dna-sim" style="width:100%; height:520px;"></div>
        <script src="https://cdn.jsdelivr.net/npm/three@0.152.2/build/three.min.js"></script>
        <script>
const dnaSeq = "{dna_sequence}";
const grnaSeq = "{gRNA}";
const targetStart = {start};
const targetEnd = {end};
const cas9Type = "{cas9_option}";

// ====================== SCENE ======================
const scene = new THREE.Scene();
scene.fog = new THREE.Fog(0x000000, 20, 120);

const container = document.getElementById("dna-sim");
const camera = new THREE.PerspectiveCamera(
    65,
    container.clientWidth / container.clientHeight,
    0.1,
    1000
);
camera.position.set(0, 0, 60);
camera.lookAt(0, 0, 0);

const renderer = new THREE.WebGLRenderer({{antialias: true, alpha: true}});
renderer.setSize(container.clientWidth, container.clientHeight);
container.appendChild(renderer.domElement);

scene.add(new THREE.AmbientLight(0xffffff, 0.7));
const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
dirLight.position.set(20, 30, 50);
scene.add(dirLight);

// ====================== UTILS ======================
function complement(b) {{
    return {{ A: "T", T: "A", C: "G", G: "C" }}[b];
}}

function colorBase(b) {{
    return {{
        A: 0xff6ec7,  // pink
        T: 0x00bfff,  // blue
        C: 0xffff00,  // yellow
        G: 0x8a2be2,  // purple
        U: 0xffa500   // orange for uracil
    }}[b];
}}
function makeLabel(text, color="#ffffff") {{
    const canvas = document.createElement("canvas");
    canvas.width = 128;
    canvas.height = 128;
    const ctx = canvas.getContext("2d");

    ctx.fillStyle = color;
    ctx.font = "bold 64px Arial";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(text, 64, 64);

    const texture = new THREE.CanvasTexture(canvas);
    return new THREE.Sprite(
        new THREE.SpriteMaterial({{ map: texture, transparent: true }})
    );
}}


function makeTitle(text, color="#00ffcc") {{
    const canvas = document.createElement("canvas");
    canvas.width = 512;
    canvas.height = 128;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = color;
    ctx.font = "bold 64px Arial";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(text, 256, 64);
    const tex = new THREE.CanvasTexture(canvas);
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({{ map: tex }}));
    sprite.scale.set(12, 3, 1);
    return sprite;
}}

// ====================== REFERENCE (NATIVE) DNA ======================
const refStrandA = new THREE.Group();
const refStrandB = new THREE.Group();
scene.add(refStrandA);
scene.add(refStrandB);

// shift native DNA to the LEFT
refStrandA.position.x = -18;
refStrandB.position.x = -18;

for (let i = 0; i < dnaSeq.length; i++) {{
    const y = i - dnaSeq.length / 2;
    const angle = i * 0.35;

    const base1 = dnaSeq[i];
    const base2 = complement(base1);

    const geo = new THREE.SphereGeometry(0.4, 14, 14);

    const s1 = new THREE.Mesh(
        geo,
        new THREE.MeshStandardMaterial({{
            color: colorBase(base1),
            transparent: true,
            opacity: 0.45
        }})
    );

    const s2 = new THREE.Mesh(
        geo,
        new THREE.MeshStandardMaterial({{
            color: colorBase(base2),
            transparent: true,
            opacity: 0.45
        }})
    );

    s1.position.set(Math.cos(angle) * 5, y, Math.sin(angle) * 5);
    s2.position.set(Math.cos(angle + Math.PI) * 5, y, Math.sin(angle + Math.PI) * 5);

    // nucleotide labels using YOUR makeLabel(text, color)
    const l1 = makeLabel(base1, "#145214"); // forest green
    l1.scale.set(0.6, 0.6, 0.6);
    s1.add(l1);

    const l2 = makeLabel(base2, "#145214");
    l2.scale.set(0.6, 0.6, 0.6);
    s2.add(l2);

    refStrandA.add(s1);
    refStrandB.add(s2);
}}

// ====================== DNA STRANDS ======================
const strandA = new THREE.Group();
const strandB = new THREE.Group();
scene.add(strandA);
scene.add(strandB);

for (let i = 0; i < dnaSeq.length; i++) {{
    const y = i - dnaSeq.length / 2;
    const angle = i * 0.35;

    const base1 = dnaSeq[i];
    const base2 = complement(base1);
    const isTarget = i >= targetStart && i < targetEnd;

    const geo = new THREE.SphereGeometry(0.45, 16, 16);

    // Strand A sphere
    const s1 = new THREE.Mesh(
        geo,
        new THREE.MeshStandardMaterial({{
            color: colorBase(base1),
            emissive: isTarget ? 0xff0044 : 0x000000,
            opacity: isTarget ? 1.0 : 0.3,
            transparent: true
        }})
    );

    // Strand B sphere
    const s2 = new THREE.Mesh(
        geo,
        new THREE.MeshStandardMaterial({{
            color: colorBase(base2),
            opacity: isTarget ? 1.0 : 0.3,
            transparent: true
        }})
    );

    s1.position.set(Math.cos(angle) * 6, y, Math.sin(angle) * 6);
    s2.position.set(Math.cos(angle + Math.PI) * 6, y, Math.sin(angle + Math.PI) * 6);

    // =========================
    // Add labels on spheres (forest green)
    // =========================
    const labelA = makeLabel(base1, "#014421");  // forest green text
    labelA.scale.set(0.8, 0.8, 0.8);
    labelA.position.set(0, 0, 0);
    labelA.material.depthTest = false;
    labelA.renderOrder = 1;
    s1.add(labelA);

    const labelB = makeLabel(base2, "#014421");  // forest green text
    labelB.scale.set(0.8, 0.8, 0.8);
    labelB.position.set(0, 0, 0);
    labelB.material.depthTest = false;
    labelB.renderOrder = 1;
    s2.add(labelB);

    // Add spheres to strands
    strandA.add(s1);
    strandB.add(s2);
}}
const nativeTitle = makeTitle("Native DNA", "#00ff99");
nativeTitle.position.set(-18, dnaSeq.length / 2 + 3, 0);
scene.add(nativeTitle);

const targetTitle = makeTitle("Target DNA (Cas9 Binding)", "#ff6ec7");
targetTitle.position.set(0, dnaSeq.length / 2 + 3, 0);
scene.add(targetTitle);

// ====================== PAM HIGHLIGHTS ======================
const pamPositions = [];

for (let i = 0; i < dnaSeq.length - 2; i++) {{
    if (cas9Type.includes("Sp") && dnaSeq.slice(i,i+3) === "GG")
        pamPositions.push(i - dnaSeq.length/2);

    else if (cas9Type.includes("Sa") && /[ACGT][ACGT]G[AG]R[AT]/.test(dnaSeq.slice(i,i+6)))
        pamPositions.push(i - dnaSeq.length/2);

    else if (cas9Type.includes("St") && /[ACGT][ACGT]AGAA[ATW]/.test(dnaSeq.slice(i,i+6)))
        pamPositions.push(i - dnaSeq.length/2);
}}

pamPositions.forEach(y => {{
    const ring = new THREE.Mesh(
        new THREE.TorusGeometry(6, 0.15, 16, 100),
        new THREE.MeshStandardMaterial({{ color: 0x00ff00, emissive: 0x00ff00 }})
    );
    ring.rotation.x = Math.PI/2;
    ring.position.y = y;
    scene.add(ring);
}});
// ====================== Cas9 COLOR BY TYPE ======================
function cas9Color(type) {{
    if (type.includes("SpCas9")) return 0x00ffff;   // cyan (classic SpCas9)
    if (type.includes("SaCas9")) return 0xff9f1c;   // amber/orange (compact SaCas9)
    if (type.includes("StCas9")) return 0x9b5de5;   // violet (thermophilic StCas9)
    return 0xffffff;
}}


// ====================== gRNA + Cas9 ======================
const grnaGroup = new THREE.Group();

grnaSeq.split("").forEach((b, i) => {{
    // Transparent casing
    const casing = new THREE.Mesh(
        new THREE.SphereGeometry(0.55, 24, 24),
        new THREE.MeshStandardMaterial({{
            color: colorBase(b),
            opacity: 0.25
        }})
    );

    // Inner nucleotide sphere
    const nt = new THREE.Mesh(
        new THREE.SphereGeometry(0.32, 16, 16),
        new THREE.MeshStandardMaterial({{
            color: colorBase(b),
            emissive: colorBase(b),
            emissiveIntensity: 0.35
        }})
    );

    // Adjust spacing dynamically
    const totalHeight = dnaSeq.length * 0.9;  // tweak if needed
    const spacing = totalHeight / grnaSeq.length;

    casing.position.set(40, i * spacing - totalHeight/2, 0);

    // Label inside casing
    const label = makeLabel(b, "#014421");  // dark forest green
    label.scale.set(1.2, 1.2, 1.2);
    label.position.set(0, 0, 0);
    label.material.depthTest = false;
    label.renderOrder = 1;

    casing.add(nt);
    casing.add(label);

    grnaGroup.add(casing);
}});

// Cas9 at 3′ end
const cas9 = new THREE.Mesh(
    new THREE.SphereGeometry(7, 48, 48),
    new THREE.MeshStandardMaterial({{
        color: cas9Color(cas9Type),
        emissive: cas9Color(cas9Type),
        emissiveIntensity: 0.35,
        transparent: true,
        opacity: 0.4
    }})
);



// Position Cas9 at PAM site (PAM-proximal end of gRNA)
const pamY = targetEnd - dnaSeq.length / 2;

cas9.position.set(
    0,          // between strands
    pamY + 1.8, // slight offset toward PAM
    0
);

grnaGroup.add(cas9);

scene.add(grnaGroup);

let unwind = 0;
let bind = 0;
const bindY = (targetStart + targetEnd) / 2 - dnaSeq.length / 2;

function animate() {{
    requestAnimationFrame(animate);

    // gentle global motion
    strandA.rotation.y += 0.005;
    strandB.rotation.y += 0.005;

    // keep native DNA always helical
    refStrandA.rotation.y += 0.005;
    refStrandB.rotation.y += 0.005;


    // ===============================
    // Phase 1: Helix → straight + separated strands
    // ===============================
    if (unwind < 1) {{
        unwind += 0.01;

        strandA.children.forEach((b, i) => {{
                b.position.x = THREE.MathUtils.lerp(b.position.x, -3.5, 0.08);
                b.position.z *= 0.90;
        }});

        strandB.children.forEach((b, i) => {{
                b.position.x = THREE.MathUtils.lerp(b.position.x, 3.5, 0.08);
                b.position.z *= 0.90;
        }});
    }}

   // ===============================
// Phase 2: gRNA inserts BETWEEN separated strands (FULL LENGTH)
// ===============================
if (unwind > 0.9 && bind < 1) {{
    cas9.material.emissiveIntensity = 0.3 + bind * 0.7;
    bind += 0.015;

    let ntIndex = 0; // real nucleotide counter (ignores "-" sprites)

    grnaGroup.children.forEach((obj) => {{
        // only move nucleotide casings (they have geometry + children)
        if (!obj.geometry) return;

        const dnaIndex = targetStart + ntIndex;
        if (dnaIndex >= targetEnd) return;

        const y = dnaIndex - dnaSeq.length / 2;

        obj.position.x += (0 - obj.position.x) * 0.15;
        obj.position.y += (y - obj.position.y) * 0.15;
        obj.position.z += (0 - obj.position.z) * 0.15;

        ntIndex++; // advance ONLY when a nucleotide is placed
    }});

    cas9.scale.set(
        1 - bind * 0.45,
        1 - bind * 0.30,
        1 - bind * 0.45
    );

    // CAMERA: center BOTH native + target DNA
    const centerX = -9; // midpoint between native (-18) and target (0)

    camera.position.z += (28 - camera.position.z) * 0.06;
    camera.position.y += (bindY - camera.position.y) * 0.06;
    camera.position.x += (centerX - camera.position.x) * 0.06;

    camera.lookAt(centerX, bindY, 0);

}}


    renderer.render(scene, camera);
}}

animate();

</script>
"""
        components.html(html, height=520)
        st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# FOOTER
# =========================================================
st.markdown(
"""
<!-- ====================== FOOTER ====================== -->
<footer style="
    margin-top:40px;
    padding:14px;
    text-align:center;
    font-size:13px;
    color:#cfcfcf;
    opacity:0.85;
">
    © 2026 Kevin Raj S · In-Silico CRISPR-Cas9 gRNA Designer
</footer>
""",
unsafe_allow_html=True
)
