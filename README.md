# DRIPS: Domain Randomization for Image-based PVS Segmentation

**DRIPS** is a method for **perivascular space (PVS) segmentation** in brain MRI volumes (NIfTI format).  
It works with both **T1-weighted** and **T2-weighted** images.  

---

## Repository Structure

- **`requirements.txt`** → lists all required Python packages.  
- **`Prediction.py`** → main script to apply segmentation to your own MRI volumes.  
- **`ext/` folder** → contains supporting files needed for inference.  

---

## How to Deploy

1. **Download DRIPS repository** (this repo).  
   - Clone it with Git:  
     ```bash
     git clone https://github.com/LunaBitar1998/DRIPS-Domain-Randomization-Image-based-PVS-Segmentation-.git
     cd DRIPS-Domain-Randomization-Image-based-PVS-Segmentation-
     ```
   - Or download the `.zip` directly from GitHub and extract it.  

2. **Download trained model and example data** from the [Releases page](https://github.com/LunaBitar1998/DRIPS-Domain-Randomization-Image-based-PVS-Segmentation-/releases/tag/v1.0.0).  
   - `Trained_Model.h5` → model weights.  
   - `Image.zip` → (optional) example MRI volume to test.  

3. **Create and activate an environment** (recommended with Python 3.8):  
   ```bash
   conda create -n drips python=3.8 -y
   conda activate drips
   ```

4. **Install dependencies**:  
   ```bash
   pip install -r requirements.txt
   ```

5. **Update paths in `Prediction.py`**:  
   - Set `path_model` to the trained model (`.h5`).  
   - Set `folder_path` to the folder containing your MRI images.  

6. **Run the prediction**:  
   ```bash
   python Prediction.py
   ```

7. **Outputs generated** in the same folder as your input image:  
   - `*_segmented.nii.gz` → segmentation map (includes all brain structures; **PVS is label 6**).  
   - `*_posteriors.nii.gz` → voxel-wise probability map for PVS (can be thresholded to make a binary mask).  
       

---

## Citation

If you use **DRIPS** in your work, please cite:  

- **SynthSeg (Benjamin Billot et al.)**: [SynthSeg papers](https://github.com/BBillot/SynthSeg)  
- **DRIPS repository**: [GitHub link](https://github.com/LunaBitar1998/DRIPS-Domain-Randomization-Image-based-PVS-Segmentation-)  

---

## License

This project is licensed under the **Apache License 2.0**.
