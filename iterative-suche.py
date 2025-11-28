# canvas_iterative_deconv_v3.py
import streamlit as st
import numpy as np
import cv2
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
import pandas as pd
from skimage.measure import label, regionprops

st.set_page_config(page_title="Iterative Kern-Zählung (OD + Deconv) — V3", layout="wide")
st.title("🧬 Iterative Kern-Zählung — V.3 (Morphologie + Kalibrierpunkt)")

# -------------------- Hilfsfunktionen --------------------
def is_near(p1, p2, r=6):
    return np.linalg.norm(np.array(p1) - np.array(p2)) < r

def dedup_new_points(candidates, existing, min_dist=6):
    """Return candidates not within min_dist of existing points."""
    out = []
    for c in candidates:
        if not any(is_near(c, e, min_dist) for e in existing):
            out.append(c)
    return out

def extract_patch(img, x, y, radius=5):
    y_min = max(0, y - radius)
    y_max = min(img.shape[0], y + radius + 1)
    x_min = max(0, x - radius)
    x_max = min(img.shape[1], x + radius + 1)
    return img[y_min:y_max, x_min:x_max]

def median_od_vector_from_patch(patch, eps=1e-6):
    if patch is None or patch.size == 0:
        return None
    OD = -np.log(np.clip((patch.astype(np.float32)+eps)/255.0, 1e-8, 1.0))
    vec = np.median(OD.reshape(-1,3), axis=0)
    norm = np.linalg.norm(vec)
    if norm < 1e-8 or np.any(np.isnan(vec)):
        return None
    return (vec / norm).astype(np.float32)

def normalize_vector(v):
    n = np.linalg.norm(v)
    return (v/n).astype(float) if n>1e-12 else v

def make_stain_matrix(target_vec, hema_vec, bg_vec=None):
    t = normalize_vector(target_vec)
    h = normalize_vector(hema_vec)
    if bg_vec is None:
        bg = np.cross(t,h)
        if np.linalg.norm(bg)<1e-6:
            bg = np.array([t[1], -t[0], 0.0], dtype=float) if abs(t[0])>0.1 or abs(t[1])>0.1 else np.array([0.0,t[2],-t[1]],dtype=float)
        bg = normalize_vector(bg)
    else:
        bg = normalize_vector(bg_vec)
    M = np.column_stack([t,h,bg]).astype(np.float32)
    return M + np.eye(3,dtype=np.float32)*1e-8

def deconvolve(img_rgb, M):
    img = img_rgb.astype(np.float32)
    OD = -np.log(np.clip((img+1e-6)/255.0,1e-8,1.0)).reshape(-1,3)
    C = (np.linalg.pinv(M) @ OD.T).T
    return C.reshape(img_rgb.shape)

def detect_centers_from_channel(channel, threshold=0.2, min_area=8, max_area=500, min_circularity=0.5, max_circularity=1.2):
    """Detect objects and apply area + circularity filter."""
    arr = np.maximum(channel.astype(np.float32),0.0)
    vmin,vmax = np.percentile(arr,[2,99.5])
    if vmax-vmin < 1e-5:
        return [], np.zeros_like(arr,dtype=np.uint8)
    norm = np.clip((arr-vmin)/(vmax-vmin),0,1)
    u8 = (norm*255).astype(np.uint8)
    blur = cv2.GaussianBlur(u8,(5,5),0)
    blockSize = 35 if min(arr.shape)>=100 else 15
    blockSize += blockSize%2==0
    mask = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize,-2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
    contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for c in contours:
        area = cv2.contourArea(c)
        if area<min_area or area>max_area:
            continue
        perimeter = cv2.arcLength(c,True)
        circ = 4*np.pi*area/(perimeter**2+1e-8)
        if circ<min_circularity or circ>max_circularity:
            continue
        M = cv2.moments(c)
        if M["m00"]!=0:
            cx = int(round(M["m10"]/M["m00"]))
            cy = int(round(M["m01"]/M["m00"]))
            centers.append((cx,cy))
    return centers, mask

# -------------------- Session State --------------------
for k in ["groups","all_points","last_file","disp_width","C_cache","last_M_hash","history"]:
    if k not in st.session_state:
        st.session_state[k] = [] if k in ["groups","all_points","history"] else None
if "disp_width" not in st.session_state: st.session_state["disp_width"] = 1000

# -------------------- UI: Upload + Parameter --------------------
uploaded_file = st.file_uploader("Bild hochladen (jpg/png/tif)", type=["jpg","png","tif","tiff"])
if not uploaded_file:
    st.info("Bitte zuerst ein Bild hochladen.")
    st.stop()

if uploaded_file.name != st.session_state.last_file:
    st.session_state.groups=[]
    st.session_state.all_points=[]
    st.session_state.C_cache=None
    st.session_state.last_M_hash=None
    st.session_state.history=[]
    st.session_state.last_file = uploaded_file.name

col1,col2 = st.columns([2,1])
with col2:
    st.sidebar.markdown("### Parameter")
    calib_radius = st.sidebar.slider("Kalibrier-Radius (px)",1,30,5)
    detection_threshold = st.sidebar.slider("Initial Threshold",0.01,0.9,0.2,0.01)
    min_area_display = st.sidebar.number_input("Min. Fläche (px, Display)",1,2000,80)
    max_area_display = st.sidebar.number_input("Max. Fläche (px, Display)",10,5000,1000)
    dedup_dist_display = st.sidebar.slider("Dedup Distanz (px, Display)",1,40,10)
    circle_radius = st.sidebar.slider("Marker-Radius (px)",1,12,5)
    min_circularity = st.sidebar.slider("Min. Rundheit",0.1,1.0,0.5,0.05)
    max_circularity = st.sidebar.slider("Max. Rundheit",0.1,1.5,1.2,0.05)
    st.sidebar.markdown("### Startvektoren (optional)")
    hema_default = st.sidebar.text_input("Hematoxylin (R,G,B)","0.65,0.70,0.29")
    aec_default = st.sidebar.text_input("Chromogen (R,G,B)","0.27,0.57,0.78")
    try:
        hema_vec0 = np.array([float(x.strip()) for x in hema_default.split(",")],dtype=float)
        aec_vec0 = np.array([float(x.strip()) for x in aec_default.split(",")],dtype=float)
    except:
        hema_vec0 = np.array([0.65,0.70,0.29])
        aec_vec0 = np.array([0.27,0.57,0.78])

with col1:
    DISPLAY_WIDTH = st.slider("Anzeige-Breite (px)",300,1600,st.session_state.disp_width)
    st.session_state.disp_width = DISPLAY_WIDTH

# -------------------- Prepare images --------------------
image_orig = np.array(Image.open(uploaded_file).convert("RGB"))
H_orig,W_orig = image_orig.shape[:2]
scale = DISPLAY_WIDTH/float(W_orig)
H_disp = int(round(H_orig*scale))
image_disp = cv2.resize(image_orig,(DISPLAY_WIDTH,H_disp),interpolation=cv2.INTER_AREA)

area_scale = (1/(scale*scale)) if scale>0 else 1
min_area_orig = max(1,int(round(min_area_display*area_scale)))
max_area_orig = max(min_area_orig,int(round(max_area_display*area_scale)))
dedup_dist_orig = max(1.0,float(dedup_dist_display/scale))

# -------------------- Draw existing points --------------------
display_canvas = image_disp.copy()
PRESET_COLORS = [(220,20,60),(0,128,0),(30,144,255),(255,165,0),(148,0,211),(0,255,255)]
for i,g in enumerate(st.session_state.groups):
    col = tuple(int(x) for x in g.get("color",PRESET_COLORS[i%len(PRESET_COLORS)]))
    for x_orig,y_orig in g["points"]:
        x_disp=int(round(x_orig*scale))
        y_disp=int(round(y_orig*scale))
        cv2.circle(display_canvas,(x_disp,y_disp),circle_radius,col,-1)
    if g["points"]:
        px_disp=int(round(g["points"][0][0]*scale))
        py_disp=int(round(g["points"][0][1]*scale))
        cv2.putText(display_canvas,f"G{i+1}:{len(g['points'])}",(px_disp+6,py_disp-6),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,col,1,cv2.LINE_AA)

coords = streamlit_image_coordinates(Image.fromarray(display_canvas),
                                    key=f"clickable_image_v3_{st.session_state.last_file}",
                                    width=DISPLAY_WIDTH)

# -------------------- Sidebar Aktionen --------------------
mode = st.sidebar.radio("Aktion", ["Kalibriere und zähle Gruppe","Punkt löschen","Undo letzte Aktion"])
st.sidebar.markdown("---")
if st.sidebar.button("Reset (Alle Gruppen)"):
    st.session_state.history.append(("reset",{
        "groups":st.session_state.groups.copy(),
        "all_points":st.session_state.all_points.copy()
    }))
    st.session_state.groups=[]
    st.session_state.all_points=[]
    st.session_state.C_cache=None
    st.success("Zurückgesetzt.")

# -------------------- Click Handling --------------------
if coords:
    x_disp,y_disp=int(coords["x"]),int(coords["y"])
    x_orig=int(round(x_disp/scale))
    y_orig=int(round(y_disp/scale))

    if mode=="Punkt löschen":
        removed=[]
        new_all=[]
        for p in st.session_state.all_points:
            if is_near(p,(x_orig,y_orig),dedup_dist_orig):
                removed.append(p)
            else:
                new_all.append(p)
        if removed:
            st.session_state.history.append(("delete_points",{"removed":removed}))
            st.session_state.all_points=new_all
            for g in st.session_state.groups:
                g["points"]=[p for p in g["points"] if not is_near(p,(x_orig,y_orig),dedup_dist_orig)]
            st.success(f"{len(removed)} Punkt(e) gelöscht.")
        else:
            st.info("Kein Punkt in der Nähe gefunden.")
    elif mode=="Undo letzte Aktion":
        if st.session_state.history:
            action,payload=st.session_state.history.pop()
            if action=="add_group":
                idx=payload["group_idx"]
                if 0<=idx<len(st.session_state.groups):
                    grp=st.session_state.groups.pop(idx)
                    for pt in grp["points"]:
                        st.session_state.all_points=[p for p in st.session_state.all_points if p!=pt]
                    st.success("Letzte Gruppen-Aktion rückgängig gemacht.")
                else:
                    st.warning("Letzte Aktion konnte nicht rückgängig gemacht werden.")
            elif action=="delete_points":
                removed=payload["removed"]
                st.session_state.all_points.extend(removed)
                st.session_state.groups.append({"vec":None,"points":removed,"color":PRESET_COLORS[len(st.session_state.groups)%len(PRESET_COLORS)]})
                st.success("Gelöschte Punkte wiederhergestellt (als neue Gruppe).")
            elif action=="reset":
                st.session_state.groups=payload["groups"]
                st.session_state.all_points=payload["all_points"]
                st.success("Reset rückgängig gemacht.")
            else:
                st.warning("Undo: unbekannte Aktion.")
        else:
            st.info("Keine Aktion zum Rückgängig machen.")
    else:
        # Kalibriere und zähle Gruppe
        patch=extract_patch(image_orig,x_orig,y_orig,calib_radius)
        vec=median_od_vector_from_patch(patch)
        if vec is None:
            st.warning("Patch unbrauchbar. Bitte anders klicken.")
        else:
            M=make_stain_matrix(vec,hema_vec0)
            M_hash=tuple(np.round(M.flatten(),6).tolist())
            recompute=False
            if st.session_state.C_cache is None or st.session_state.last_M_hash!=M_hash:
                recompute=True
            if recompute:
                C_full=deconvolve(image_orig,M)
                st.session_state.C_cache=C_full
                st.session_state.last_M_hash=M_hash
            else:
                C_full=st.session_state.C_cache
            channel_full=C_full[:,:,0]

            # Detektion mit Morphologie
            centers_orig,mask = detect_centers_from_channel(channel_full,
                                                            threshold=detection_threshold,
                                                            min_area=min_area_orig,
                                                            max_area=max_area_orig,
                                                            min_circularity=min_circularity,
                                                            max_circularity=max_circularity)
            # Kalibrierpunkt hinzufügen
            if (x_orig,y_orig) not in centers_orig:
                centers_orig.append((x_orig,y_orig))

            new_centers=dedup_new_points(centers_orig,st.session_state.all_points,min_dist=dedup_dist_orig)
            if new_centers:
                color=PRESET_COLORS[len(st.session_state.groups)%len(PRESET_COLORS)]
                group={"vec":vec.tolist(),"points":new_centers,"color":color}
                st.session_state.history.append(("add_group",{"group_idx":len(st.session_state.groups)}))
                st.session_state.groups.append(group)
                st.session_state.all_points.extend(new_centers)
                st.success(f"Gruppe hinzugefügt — neue Kerne: {len(new_centers)}")
            else:
                st.info("Keine neuen Kerne (alle bereits gezählt oder keine Detektion).")

# -------------------- Ergebnis & Export --------------------
st.markdown("## Ergebnisse")
colA,colB = st.columns([2,1])
with colA:
    st.image(display_canvas, caption="Gezählte Kerne", use_column_width=True)

with colB:
    st.markdown("### Zusammenfassung")
    st.write(f"🔹 Gruppen gesamt: {len(st.session_state.groups)}")
    for i,g in enumerate(st.session_state.groups):
        st.write(f"• Gruppe {i+1}: {len(g['points'])} Kerne")
    st.markdown(f"**Gesamt (unique Kerne): {len(st.session_state.all_points)}**")

    if st.session_state.all_points:
        rows=[]
        for i,g in enumerate(st.session_state.groups):
            for x_orig,y_orig in g["points"]:
                x_disp=int(round(x_orig*scale))
                y_disp=int(round(y_orig*scale))
                rows.append({"Group":i+1,"X_display":x_disp,"Y_display":y_disp,
                             "X_original":x_orig,"Y_original":y_orig})
        df=pd.DataFrame(rows)
        st.download_button("📥 CSV exportieren (Gruppen)", df.to_csv(index=False).encode("utf-8"),
                           file_name="kern_gruppen_v3.csv", mime="text/csv")
