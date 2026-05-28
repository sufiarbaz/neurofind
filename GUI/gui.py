import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import numpy as np
import tifffile
import cv2
import torch
import matplotlib.pyplot as plt


from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QSlider, QFileDialog, QGraphicsView, QGraphicsScene
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QPen, QColor
import pyqtgraph as pg
from utils import get_points_stack2, crop_around_point, compute_embedding, compare_embeddings, to_uint8, DINOv3Encoder, ResNet3DEncoder, DINO3DEncoder
from tqdm import tqdm

# ----------------------------
# Image viewer widget
# ----------------------------

class ImageStackViewer(QWidget):
    def __init__(self):
        super().__init__()

        self.layout = QVBoxLayout(self)
        self.image_view = pg.ImageView()
        self.layout.addWidget(self.image_view)

        self.stack = None
        self.click_enabled = False
        self.click_callback = None

        # Connect mouse click
        self.image_view.getView().scene().sigMouseClicked.connect(self._mouse_clicked)

    def load_stack(self, stack):
        """
        stack: numpy array [Z, H, W]
        """
        # reshape to (Z, W, H)
        stack = np.transpose(stack, (0, 2, 1))
        self.stack = stack
        self.image_view.setImage(stack)
        # lock base image to grayscale
        self.image_view.setColorMap(pg.ColorMap(
            [0.0, 1.0],
            [[0, 0, 0], [255, 255, 255]]
        ))

    def show_image(self, img):
        """
        img: 2D or RGB image
        """
        self.image_view.setImage(img)

    def _mouse_clicked(self, event):
        if not self.click_enabled or self.click_callback is None:
            return

        pos = event.scenePos()
        mouse_point = self.image_view.getView().mapSceneToView(pos)

        x = int(mouse_point.x())
        y = int(mouse_point.y())

        # get current Z index
        z = self.image_view.currentIndex

        self.click_callback(x, y, z)
        self.click_enabled = False

    def draw_point(self, x, y, color=(255, 0, 0), size=6):
        scatter = pg.ScatterPlotItem(
            [x], [y],
            pen=pg.mkPen(color=color, width=2),
            brush=None,
            size=size
        )
        self.image_view.addItem(scatter)


# ----------------------------
# Main window
# ----------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stack Matching GUI")


        self.model1 = None
        self.model2 = None
        self.model3 = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.stack1 = None
        self.stack2 = None

        self.match_sims = None        # np.ndarray [N_points]
        self.best_match_idx = None

        self.points_stack2 = []
        self.embeddings_stack2_model1 = []
        self.embeddings_stack2_model2 = []
        self.embeddings_stack2_model3 = []

        self.viewer1 = ImageStackViewer()
        self.viewer2 = ImageStackViewer()

        self.overlay_img = pg.ImageItem()
        self.overlay_img.setZValue(10)  # draw above base image
        self.viewer2.image_view.addItem(self.overlay_img)

        self.select_btn = QPushButton("Select point")

        # self.sim_cmap = pg.colormap.get("magma")  # or viridis, inferno, etc.
        # self.sim_colorbar = pg.ColorBarItem(
        #     values=(0, 1),
        #     colorMap=self.sim_cmap
        # )

        # self.sim_colorbar.setLevels((0, 1))
        # self.sim_colorbar.levels = (0, 1)  # hard override
        # self.sim_colorbar.vb.setMouseEnabled(False, False)

        # self.sim_colorbar.setImageItem(self.overlay_img)
        # self.viewer2.image_view.addItem(self.sim_colorbar)
        self.sim_lut = pg.HistogramLUTItem()
        self.sim_lut.setImageItem(self.overlay_img)
        self.sim_lut.setLevels(0, 1)
        self.sim_lut.disableAutoHistogramRange()

        # Disable vertical scaling interaction
        self.sim_lut.vb.setMouseEnabled(x=False, y=False)
        self.sim_lut.vb.setMenuEnabled(False)
        self.sim_lut.plot.setVisible(False)

        self.viewer2.image_view.addItem(self.sim_lut)
        self.sim_lut.gradient.loadPreset('magma')


        self.viewer2.image_view.sigTimeChanged.connect(
            lambda: self.update_stack2_overlay()
        )
        self.sim_lut.gradient.sigGradientChanged.connect(
            self.update_stack2_overlay
        )

        self._layout()
        self._connect()

    def _layout(self):
        central = QWidget()
        self.setCentralWidget(central)

        v1 = QVBoxLayout()
        v1.addWidget(self.viewer1)

        v2 = QVBoxLayout()
        v2.addWidget(self.viewer2)

        h = QHBoxLayout()
        h.addLayout(v1)
        h.addLayout(v2)

        main = QVBoxLayout()
        main.addLayout(h)
        main.addWidget(self.select_btn)

        central.setLayout(main)

    def _connect(self):
        self.select_btn.clicked.connect(self.enable_point_selection)

    # ----------------------------
    # Load stacks
    # ----------------------------
    def load_model(self, model_path1, model_path2, model_path3):

        self.model1 = DINOv3Encoder(embedding_dim=128, model='dinov3_vitb16').to(self.device) # dinov3
        self.model2 = ResNet3DEncoder(embedding_dim=128, use_multichannel=False).to(self.device)  # resnet3d
        self.model3 = DINO3DEncoder(embedding_dim=128, freeze_backbone=False).to(self.device) # 3DINO
        self.model1.load_state_dict(torch.load(model_path1, map_location=self.device))
        self.model2.load_state_dict(torch.load(model_path2, map_location=self.device))
        self.model3.load_state_dict(torch.load(model_path3, map_location=self.device))

    def load_stack1(self, path):
        self.stack1 = tifffile.imread(path)
        self.viewer1.load_stack(self.stack1)

    def load_stack2(self, path, embedding_path1, embedding_path2, embedding_path3):
        self.stack2 = tifffile.imread(path)
        self.viewer2.load_stack(self.stack2)

        self.points_stack2 = get_points_stack2(self.stack2)

        # File where embeddings will be cached
        for [emb_path, model] in [[embedding_path1, self.model1], [embedding_path2, self.model2], [embedding_path3, self.model3]]:

            if os.path.exists(emb_path):
                print("Loading cached embeddings...")
                emb = np.load(emb_path)
                # store embeddings in the correct variable
                if model == self.model1:
                    self.embeddings_stack2_model1 = emb
                elif model == self.model2:
                    self.embeddings_stack2_model2 = emb
                elif model == self.model3:
                    self.embeddings_stack2_model3 = emb
            else:
                print("Computing embeddings...")
                emb_list = [
                    compute_embedding(
                        crop_around_point(self.stack2, p),
                        model,
                        self.device
                    )
                    for p in tqdm(self.points_stack2, desc="Computing embeddings")
                ]

                emb = np.stack(emb_list)  # shape: (N, D)
                # store embeddings in the correct variable
                if model == self.model1:
                    self.embeddings_stack2_model1 = emb
                elif model == self.model2:
                    self.embeddings_stack2_model2 = emb
                elif model == self.model3:
                    self.embeddings_stack2_model3 = emb

                print("Saving embeddings...")
                np.save(emb_path, emb)
    

    def update_stack2_overlay(self):

        z = self.viewer2.image_view.currentIndex

        if self.match_sims is None:
            self.overlay_img.clear()
            return

        points = np.asarray(self.points_stack2)
        sims = np.asarray(self.match_sims)

        slice_mask = points[:, 0] == z
        slice_points = points[slice_mask]
        # slice_sims = sims[slice_mask]

        h, w = self.stack2.shape[1:]

        # Create empty transparent RGBA overlay
        overlay = np.zeros((h, w, 4), dtype=np.float16)

        if len(slice_points) == 0:
            self.overlay_img.setImage(overlay)
            return

        # ---- rank normalization ----
        if len(sims) > 1:
            ranks = np.argsort(np.argsort(sims))
            sims_norm = ranks / (len(sims) - 1)
        else:
            sims_norm = np.array([1.0])
        slice_sims = sims_norm[slice_mask]
        # print(f"Slice {z}: {len(slice_points)} points, sims_norm range: {sims_norm.min():.4f} - {sims_norm.max():.4f}")
        # --- use active pyqtgraph colormap ---
        # self.sim_cmap = self.sim_colorbar.colorMap()
        cmap = self.sim_lut.gradient.colorMap()
        colors = cmap.map(slice_sims, mode='float')
        colors = (colors[:, :3]).astype(np.float16)
        # print(f"Color range for slice {z}: R={colors[:,0].min()}-{colors[:,0].max()}, G={colors[:,1].min()}-{colors[:,1].max()}, B={colors[:,2].min()}-{colors[:,2].max()}")

        xs = slice_points[:, 1].astype(int)
        ys = slice_points[:, 2].astype(int)

        valid = (
            (xs >= 0) & (xs < w) &
            (ys >= 0) & (ys < h)
        )

        xs = xs[valid]
        ys = ys[valid]
        colors = colors[valid]

        # ---- write single pixels ----
        overlay[ys, xs, :3] = colors
        overlay[ys, xs, 3] = 180  # alpha

        if self.best_match_idx is not None:

            bz, bx, by = self.points_stack2[self.best_match_idx]

            if bz == z and 0 <= bx < w and 0 <= by < h:

                r = 3
                y_min = max(0, by - r)
                y_max = min(h, by + r + 1)
                x_min = max(0, bx - r)
                x_max = min(w, bx + r + 1)

                # solid red square
                overlay[y_min:y_max, x_min:x_max, 0] = 255
                overlay[y_min:y_max, x_min:x_max, 1] = 0
                overlay[y_min:y_max, x_min:x_max, 2] = 0
                overlay[y_min:y_max, x_min:x_max, 3] = 255  # fully opaque

        self.overlay_img.setImage(overlay, autoLevels=False)

    # ----------------------------
    # Point selection & matching
    # ----------------------------

    def enable_point_selection(self):
        self.viewer1.click_enabled = True
        self.viewer1.click_callback = self.point_selected

    def point_selected(self, x, y, z):
        # visualize selected point in stack 1
        print(f"Selected point in stack1: (z={z}, y={y}, x={x})")
        # x, y, z = 314, 231, 9
        self.viewer1.draw_point(x, y, QColor(255, 0, 0))

        crop = crop_around_point(self.stack1, (z, y, x))
        query_emb1 = compute_embedding(crop, self.model1, self.device)
        query_emb2 = compute_embedding(crop, self.model2, self.device)
        query_emb3 = compute_embedding(crop, self.model3, self.device)

        _, sims1 = compare_embeddings(query_emb1, self.embeddings_stack2_model1)
        _, sims2 = compare_embeddings(query_emb2, self.embeddings_stack2_model2)
        _, sims3 = compare_embeddings(query_emb3, self.embeddings_stack2_model3)

        sims1 = (sims1 - sims1.min()) / (sims1.max() - sims1.min() + 1e-8)
        sims2 = (sims2 - sims2.min()) / (sims2.max() - sims2.min() + 1e-8)
        sims3 = (sims3 - sims3.min()) / (sims3.max() - sims3.min() + 1e-8)
        sims = (sims1 + sims2 + sims3) / 3.0

        # ---- store state (ONCE) ----
        sims = sims.cpu().numpy()
        self.match_sims = sims
        self.best_match_idx = sims.argmax()
        print(f"Best match in stack2: {self.points_stack2[self.best_match_idx]}, similarity={sims[self.best_match_idx]:.4f}")
        self.show_similarity_histogram()

        # ---- redraw current slice ----
        self.update_stack2_overlay()
    
    def show_similarity_histogram(self):
        if self.match_sims is None:
            return

        sims = np.array(self.match_sims)

        plt.figure()
        plt.hist(sims, bins=200)
        plt.title("Similarity Distribution")
        plt.xlabel("Similarity")
        plt.ylabel("Count")
        plt.show()


# ----------------------------
# Run
# ----------------------------

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()

    # Example loading
    win.load_model(
        "hpc_results/models_test/spine_embedder_ssl_dinov3_mean_64_9_5.pth",
        "hpc_results/models_test/spine_embedder_ssl_resnet_64_9_5.pth",
        "hpc_results/models_test/spine_embedder_ssl_3dino_64_9_5.pth")
    win.load_stack1("D:/jo77pihe/Original_Data_from_Alessio_and_Bhargavi/Ghabiba/25X_1NA_DeconvolvedScored/65_Thy1eGFP/2018-02-03/A1.tif")
    win.load_stack2("D:/jo77pihe/Original_Data_from_Alessio_and_Bhargavi/Ghabiba/25X_1NA_DeconvolvedScored/65_Thy1eGFP/2018-02-04/A1.tif", 
                    "GUI/embeddings/embeddings_dinov3_reflective.npy",
                    "GUI/embeddings/embeddings_resnet_reflective.npy",
                    "GUI/embeddings/embeddings_3dino_reflective.npy")

    win.show()
    sys.exit(app.exec_())
