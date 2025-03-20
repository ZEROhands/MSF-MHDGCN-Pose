class PoseDataset:
    Initialize:
        - `mode`: **"train" or "test"**
        - `num_pt`: **Number of points per object**
        - `add_noise`: **Enable noise augmentation**
        - `root`: **Dataset root directory**
        - `noise_trans`: **Translation noise range**
        - `refine`: **Whether used for pose refinement**
        - `is_spherical`: **Convert point cloud to spherical coordinates**
        
        1. **Load dataset list**
            - Open `{root}/dataset_config/train_data_list.txt` or `test_data_list.txt`
            - Read file line by line:
                - Append to `self.list`
                - If `data/`, append to `self.real`
                - Otherwise, append to `self.syn`
        
        2. **Load object point clouds**
            - Open `{root}/dataset_config/classes.txt`
            - Iterate over object classes:
                - Read `{root}/models/{class_name}/points.xyz`
                - Store 3D points in `self.cld[class_id]`
        
        3. **Initialize camera parameters**
            - Set `cam_cx, cam_cy, cam_fx, cam_fy` for two different camera settings  
        
        4. **Prepare transformations & settings**
            - `self.trancolor = ColorJitter(...)` for **color augmentation**
            - `self.norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])` for **RGB normalization**
            - `self.resize = Resize((80, 80))` for **size normalization**
            - `self.num_pt_mesh_large = 2600`, `self.num_pt_mesh_small = 500`
        
    **Load an item (`__getitem__(index)`)**:
        1. **Read image, depth, label, and metadata**
            - `img = Image.open('{root}/{self.list[index]}-color.png')`
            - `depth = np.array(Image.open('{root}/{self.list[index]}-depth.png'))`
            - `label = np.array(Image.open('{root}/{self.list[index]}-label.png'))`
            - `meta = loadmat('{root}/{self.list[index]}-meta.mat')`
        
        2. **Select camera parameters**
            - If **synthetic data**: use `cam_cx_1, cam_cy_1, cam_fx_1, cam_fy_1`
            - Else: use `cam_cx_2, cam_cy_2, cam_fx_2, cam_fy_2`
        
        3. **Segment target object**
            - `mask_depth = mask where depth != 0`
            - `mask_label = mask where label == obj[idx]`
            - `mask = mask_label * mask_depth`
            - **Ensure object has enough points (`self.minimum_num_pt`)**
        
        4. **Apply data augmentation (if `add_noise=True`)**
            - `self.trancolor(img)`: **Color jitter**
            - **Foreground augmentation**:
                - Select a random synthetic image `seed`
                - Blend foreground objects into `img`
            - **Background replacement**:
                - If synthetic data, replace background with a real image
        
        5. **Extract object region from image**
            - `rmin, rmax, cmin, cmax = get_bbox(mask_label)`
            - Crop `img[:, rmin:rmax, cmin:cmax]`
            - `img = self.resize(img)`  # **Resize to `80 × 80`**
            - `img = self.norm(img)`  # **Normalize using `mean` and `std`**
        
        6. **Compute 3D point cloud from depth**
            - `depth_masked = depth[rmin:rmax, cmin:cmax]`
            - `xmap_masked, ymap_masked = camera coordinate maps`
            - `pt2 = depth_masked / cam_scale`
            - `pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx`
            - `pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy`
            - `cloud = np.concatenate((pt0, pt1, pt2), axis=1)`
            - If `add_noise=True`: `cloud += np.random.uniform(-noise_trans, noise_trans)`
            - If `is_spherical=True`: Convert `cloud` to spherical coordinates
        
        7. **Normalize point cloud**
            - `cloud = cloud - mean(cloud, axis=0)`  # **Center the point cloud**
            - `scale_factor = max(norm(cloud, axis=1))`
            - `cloud = cloud / scale_factor`  # **Normalize to a unit scale**
        
        8. **Sample `num_pt` points**
            - `choose = select nonzero indices from mask`
            - If `len(choose) > num_pt`: randomly sample `num_pt`
            - Else: pad `choose` to `num_pt`
        
        9. **Load 3D model points**
            - `model_points = self.cld[obj[idx]]`
            - `if refine: sample num_pt_mesh_large points`
            - `else: sample num_pt_mesh_small points`
        
        10. **Compute pose transformation**
            - `target_r = meta['poses'][:, :, idx][:, 0:3]` **(Rotation matrix)**
            - `target_t = meta['poses'][:, :, idx][:, 3:4].flatten()` **(Translation vector)**
            - `target = np.dot(model_points, target_r.T) + target_t`
        
        11. **Return processed data**
            - `cloud`: **Point cloud (num_pt, 3)**
            - `R`: **Rotation matrix**
            - `choose`: **Indices of selected points**
            - `img`: **Resized & Normalized RGB image**
            - `target`: **Transformed model points**
            - `model_points`: **Original model points**
            - `obj`: **Object class ID**
        
    **Dataset length (`__len__`)**:
        - `return self.length`
    
    **Get symmetrical objects (`get_sym_list`)**:
        - `return self.symmetry_obj_idx`
    
    **Get model point count (`get_num_points_mesh`)**:
        - `return self.num_pt_mesh_large` if `refine=True`, else `self.num_pt_mesh_small`
    
**Bounding Box Extraction (`get_bbox`)**:
    - Compute `rmin, rmax, cmin, cmax` where the object exists
    - Adjust bounding box size based on predefined `border_list`
